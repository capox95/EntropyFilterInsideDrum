#include "../include/segmentation.h"

void SegFilter::setSourceCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) { source_ = cloud; }

void SegFilter::getOutputCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) { cloud = cloud_output_; }

void SegFilter::getDrumAxis(pcl::ModelCoefficients &axis) { axis = line1_; }

void SegFilter::compute()
{
    computeDrumAxes(line1_, line2_);

    // origin of the Drum
    line1_.values[0] = 0;
    line1_.values[1] = 0.09;
    line1_.values[2] = 0.11;
    line2_.values[0] = 0;
    line2_.values[1] = 0.09;
    line2_.values[2] = 0.11;

    /*
    std::cout << std::endl;
    for (int i = 0; i < line1_.values.size(); i++)
        std::cout << line1_.values[i] << ", ";
    std::cout << std::endl;
    for (int i = 0; i < line1_.values.size(); i++)
        std::cout << line2_.values[i] << ", ";
    std::cout << std::endl;
    */

    Eigen::Vector3f axis_dir = {line1_.values[3], line1_.values[4], line1_.values[5]};
    Eigen::Vector3f origin = {line1_.values[0], line1_.values[1], line1_.values[2]};

    Eigen::Vector3f center = origin + axis_dir * distanceCenterDrum_;
    centerPoint_.getVector3fMap() = center;

    segPoint1_.getVector3fMap() = center - (drumDepth_ / 2) * axis_dir;
    segPoint2_.getVector3fMap() = center + (drumDepth_ / 2) * axis_dir;

    Eigen::Vector3f vector_dir = {line2_.values[3], line2_.values[4], line2_.values[5]};

    pointsHull1_ = calculateHullPoints(segPoint1_, axis_dir, vector_dir, radiusDrum_);
    pointsHull2_ = calculateHullPoints(segPoint2_, axis_dir, vector_dir, radiusDrum_);

    combineHullPoints(pointsHull1_, pointsHull2_, hull_vertices_);

    pcl::copyPointCloud(*source_, *source_bw_);
    convexHullCrop(source_bw_, hull_vertices_, hull_result_);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);
    refineCrop(hull_result_, cloud_out, segPoint1_, axis_dir);

    pcl::copyPointCloud(*cloud_out, *cloud_output_);
}

void SegFilter::visualizeSeg(bool flagSpin)
{
    pcl::visualization::PCLVisualizer vizSource("PCL Transformation");
    vizSource.addCoordinateSystem(0.2, "coord");
    vizSource.setBackgroundColor(0.0f, 0.0f, 0.5f);
    vizSource.addPointCloud<pcl::PointXYZRGB>(source_, "source_");

    vizSource.addLine(line1_, "line1");
    vizSource.addLine(line2_, "line2");

    vizSource.addPointCloud<pcl::PointXYZ>(hull_vertices_, "hull");
    vizSource.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0f, 0.0f, 0.0f, "hull");

    if (flagSpin)
        vizSource.spin();
}

void SegFilter::transformation(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out)
{
    float theta = 0.9075; // 0.9075 --- 52 deg

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << 0.0, 0.0, 0.0;
    transform.rotate(Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitX()));
    pcl::transformPointCloud(*cloud, *cloud_out, transform);
}

void SegFilter::computeDrumAxes(pcl::ModelCoefficients &line1, pcl::ModelCoefficients &line2)
{

    //add fake points along z axis

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_fake(new pcl::PointCloud<pcl::PointXYZ>);
    int numPoints_1 = 50, numPoints_2 = 25;

    cloud_fake->width = numPoints_1 + numPoints_2;
    cloud_fake->height = 1;
    cloud_fake->resize(cloud_fake->height * cloud_fake->width);

    for (int k = 0; k < cloud_fake->size(); k++)
    {
        if (k <= numPoints_1)
        {
            cloud_fake->points[k].x = 0;
            cloud_fake->points[k].y = 0;
            cloud_fake->points[k].z = 0 + k / 100.0f;
        }
        else
        {
            cloud_fake->points[k].x = 0;
            cloud_fake->points[k].y = 0 + (k - numPoints_2) / 100.0f;
            cloud_fake->points[k].z = 0;
        }

        //std::cout << cloud_fake->points[k].getVector3fMap() << std::endl;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_fake_transformed(new pcl::PointCloud<pcl::PointXYZ>);
    transformation(cloud_fake, cloud_fake_transformed);

    // line detection
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::ExtractIndices<pcl::PointXYZ> extract;

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_LINE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.02);

    seg.setInputCloud(cloud_fake_transformed);
    seg.segment(*inliers, line1);

    extract.setInputCloud(cloud_fake_transformed);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*cloud_fake_transformed);

    seg.setInputCloud(cloud_fake_transformed);
    seg.segment(*inliers, line2);
}

std::vector<pcl::PointXYZ> SegFilter::calculateHullPoints(pcl::PointXYZ &point1, Eigen::Vector3f &axis,
                                                          Eigen::Vector3f &vector, float radius_cylinder)
{

    //extra padding
    radius_cylinder += 0.05; // <<<----------------------------------

    std::vector<pcl::PointXYZ> newPoints;
    pcl::PointXYZ new_point;

    // Rodrigues' rotation formula

    // angle to rotate
    float theta = M_PI / 10;

    // unit versor k
    Eigen::Vector3f k = axis;
    k.normalize();

    // vector to rotate V
    Eigen::Vector3f V = vector;
    Eigen::Vector3f V_rot;

    V_rot = V * cos(-M_PI_2) + (k.cross(V)) * sin(-M_PI_2) + k * (k.dot(V)) * (1 - cos(-M_PI_2));
    V_rot.normalize();
    new_point.x = point1.x + radius_cylinder * V_rot.x();
    new_point.y = point1.y + radius_cylinder * V_rot.y();
    new_point.z = point1.z + radius_cylinder * V_rot.z();

    newPoints.push_back(new_point);
    V = V_rot;

    for (int c = 0; c < 10; c++)
    {

        V_rot = V * cos(theta) + (k.cross(V)) * sin(theta) + k * (k.dot(V)) * (1 - cos(theta));
        V_rot.normalize();

        new_point.x = point1.x + radius_cylinder * V_rot.x();
        new_point.y = point1.y + radius_cylinder * V_rot.y();
        new_point.z = point1.z + radius_cylinder * V_rot.z();

        newPoints.push_back(new_point);
        V = V_rot;
    }

    newPoints.push_back(point1);

    return newPoints;
}

void SegFilter::combineHullPoints(std::vector<pcl::PointXYZ> &p1, std::vector<pcl::PointXYZ> &p2,
                                  pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    cloud->width = p1.size() + p2.size();
    cloud->height = 1;
    cloud->resize(cloud->width * cloud->height);

    for (int i = 0; i < p1.size(); i++)
        cloud->points.push_back(p1[i]);

    for (int i = 0; i < p2.size(); i++)
        cloud->points.push_back(p2[i]);
}

void SegFilter::convexHullCrop(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_bw,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_vertices,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr &hull_result)
{
    pcl::CropHull<pcl::PointXYZ> cropHullFilter;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr points_hull(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<pcl::Vertices> hullPolygons;

    // setup hull filter
    pcl::ConcaveHull<pcl::PointXYZ> cHull;
    cHull.setInputCloud(cloud_vertices);
    cHull.setDimension(3);
    cHull.setAlpha(1.0);
    cHull.reconstruct(*points_hull, hullPolygons);

    cropHullFilter.setHullIndices(hullPolygons);
    cropHullFilter.setHullCloud(points_hull);
    //cropHullFilter.setDim(3);
    cropHullFilter.setCropOutside(true);

    //filter points
    cropHullFilter.setInputCloud(cloud_bw);
    cropHullFilter.filter(*hull_result);

    //std::cout << std::endl;
    //std::cout << "hull result points: " << hull_result->points.size() << std::endl;
    //for (int i = 0; i < hull_result->points.size(); i++)
    //{
    //    std::cout << hull_result->points[i] << std::endl;
    //}
}

void SegFilter::refineCrop(pcl::PointCloud<pcl::PointXYZ>::Ptr &hull_result, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out,
                           pcl::PointXYZ &point, Eigen::Vector3f &axis)
{
    pcl::ModelCoefficients plane;
    float d = point.getVector3fMap().norm();
    plane.values = {axis.x(), axis.y(), axis.z(), -d};

    pcl::copyPointCloud(*hull_result, *cloud_out);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    Eigen::Vector3f x0;
    Eigen::Vector3f n(plane.values[0], plane.values[1], plane.values[2]);
    float p = plane.values[3];
    float distance;
    int counterp = 0, counterm = 0;

    for (int i = 0; i < cloud_out->size(); i++)
    {
        if ((!std::isnan(cloud_out->points[i].x)) && (!std::isnan(cloud_out->points[i].y)) && (!std::isnan(cloud_out->points[i].z)))
        {
            x0 = cloud_out->points[i].getVector3fMap();
            distance = n.dot(x0) + p;

            if (distance > 0)
                counterp++;
            else
            {
                counterm++;
                inliers->indices.push_back(i);
            }
        }
    }

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_out);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*cloud_out);

    //std::cout << "counter positive " << counterp << ", counter neg " << counterm << std::endl;

    /*
    pcl::visualization::PCLVisualizer viz("plane");
    viz.addCoordinateSystem(0.1, "coord");
    viz.setBackgroundColor(0.0f, 0.0f, 0.5f);
    viz.addPointCloud<pcl::PointXYZ>(cloud_out, "cloud_out");
    viz.addPlane(plane, "plane");
    viz.spin();
    */
}