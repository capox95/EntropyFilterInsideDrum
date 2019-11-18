#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/pca.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/distances.h>

#include "../include/entropy.h"

void EntropyFilter::setInputCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in) { m_source = cloud_in; }

void EntropyFilter::setDrumAxis(pcl::ModelCoefficients line) { m_line = line; }

void EntropyFilter::setDownsampleLeafSize(float leaf_size) { m_leafsize = leaf_size; }

void EntropyFilter::setEntropyThreshold(float entropy_th) { m_entropy_threshold = entropy_th; }

void EntropyFilter::setKLocalSearch(int K) { m_KNN = K; }

void EntropyFilter::setCurvatureThreshold(float curvature_th) { m_curvature_threshold = curvature_th; }

void EntropyFilter::setDepthThreshold(float depth_th) { m_depth_threshold = depth_th; };

pcl::PointCloud<pcl::PointXYZ>::Ptr EntropyFilter::getMLSCloud() { return m_mls_cloud; }

pcl::PointCloud<pcl::Normal>::Ptr EntropyFilter::getMLSNormals() { return m_mls_normals; }

bool EntropyFilter::compute(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clouds_out)
{
    //downsample(m_source, m_leafsize, m_cloud_downsample);
    computePolyFitting(m_source, m_mls_points);
    divideCloudNormals(m_mls_points, m_mls_cloud, m_mls_normals);
    getSpherical(m_mls_normals, m_spherical);

    //Depth
    computeDepthMap(m_mls_cloud, m_cloud_depth, m_line);

    //Combine
    combineDepthAndCurvatureInfo(m_cloud_depth, m_mls_normals, m_cloud_combined);

    local_search(m_mls_cloud, m_spherical, m_cloud_combined);
    normalizeEntropy(m_spherical);

    if (_max_entropy < 1.0)
    {
        PCL_WARN("Entropy too small!\n");
        return false;
    }

    segmentCloudEntropy(m_mls_points, m_spherical, m_cloud_seg, m_entropy_threshold);

    if (m_cloud_seg->size() > 50)
    {

        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds_connected;
        connectedComponets(m_cloud_seg, clouds_connected);
        clouds_out = clouds_connected;
    }
    else
        clouds_out.push_back(m_cloud_seg);

    return true;
}

//
//ColorMap functions
void EntropyFilter::colorMapEntropy(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_map)
{
    cloud_map->width = m_mls_cloud->width;
    cloud_map->height = m_mls_cloud->height;
    cloud_map->resize(m_mls_cloud->width * m_mls_cloud->height);

    for (int i = 0; i < m_mls_cloud->size(); i++)
    {
        cloud_map->points[i].x = m_mls_cloud->points[i].x;
        cloud_map->points[i].y = m_mls_cloud->points[i].y;
        cloud_map->points[i].z = m_mls_cloud->points[i].z;
        cloud_map->points[i].intensity = m_spherical->points[i].entropy;
    }
}

void EntropyFilter::colorMapCurvature(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_map)
{
    cloud_map->width = m_mls_cloud->width;
    cloud_map->height = m_mls_cloud->height;
    cloud_map->resize(m_mls_cloud->width * m_mls_cloud->height);

    for (int i = 0; i < m_mls_cloud->size(); i++)
    {
        cloud_map->points[i].x = m_mls_cloud->points[i].x;
        cloud_map->points[i].y = m_mls_cloud->points[i].y;
        cloud_map->points[i].z = m_mls_cloud->points[i].z;
        cloud_map->points[i].intensity = m_mls_normals->points[i].curvature;
    }
}

void EntropyFilter::colorMapInclination(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_map)
{
    cloud_map->width = m_mls_cloud->width;
    cloud_map->height = m_mls_cloud->height;
    cloud_map->resize(m_mls_cloud->width * m_mls_cloud->height);

    for (int i = 0; i < m_mls_cloud->size(); i++)
    {
        cloud_map->points[i].x = m_mls_cloud->points[i].x;
        cloud_map->points[i].y = m_mls_cloud->points[i].y;
        cloud_map->points[i].z = m_mls_cloud->points[i].z;
        cloud_map->points[i].intensity = m_spherical->points[i].inclination;
    }
}

void EntropyFilter::colorMapAzimuth(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_map)
{
    cloud_map->width = m_mls_cloud->width;
    cloud_map->height = m_mls_cloud->height;
    cloud_map->resize(m_mls_cloud->width * m_mls_cloud->height);

    for (int i = 0; i < m_mls_cloud->size(); i++)
    {
        cloud_map->points[i].x = m_mls_cloud->points[i].x;
        cloud_map->points[i].y = m_mls_cloud->points[i].y;
        cloud_map->points[i].z = m_mls_cloud->points[i].z;
        cloud_map->points[i].intensity = m_spherical->points[i].azimuth;
    }
}

void EntropyFilter::visualizeAll(bool flag)
{

    //pcl::visualization::PCLVisualizer vizSource("PCL Source Cloud");
    //vizSource.addCoordinateSystem(0.2, "coord");
    //vizSource.setBackgroundColor(1.0f, 1.0f, 1.0f);
    //vizSource.addPointCloud<pcl::PointXYZRGB>(m_source, "m_source");
    //vizSource.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0f, 1.0f, 0.0f, "m_source");

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_curvature(new pcl::PointCloud<pcl::PointXYZI>);
    colorMapCurvature(cloud_curvature);
    pcl::visualization::PCLVisualizer vizC("PCL Curvature Map");
    vizC.setBackgroundColor(1.0f, 1.0f, 1.0f);
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distributionCurvature(cloud_curvature, "intensity");
    vizC.addPointCloud<pcl::PointXYZI>(cloud_curvature, intensity_distributionCurvature, "cloud_mapCurvature");
    vizC.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_mapCurvature");

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_entropy(new pcl::PointCloud<pcl::PointXYZI>);
    colorMapEntropy(cloud_entropy);
    pcl::visualization::PCLVisualizer vizE("PCL Entropy Map");
    vizE.setBackgroundColor(1.0f, 1.0f, 1.0f);
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distributionEntropy(cloud_entropy, "intensity");
    vizE.addPointCloud<pcl::PointXYZI>(cloud_entropy, intensity_distributionEntropy, "cloud_mapEntropy");
    vizE.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_mapEntropy");

    if (flag)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_inclination(new pcl::PointCloud<pcl::PointXYZI>);
        colorMapInclination(cloud_inclination);
        pcl::visualization::PCLVisualizer vizI("PCL Inclination Map");
        vizI.setBackgroundColor(1.0f, 1.0f, 1.0f);
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distributionInclination(cloud_inclination, "intensity");
        vizI.addPointCloud<pcl::PointXYZI>(cloud_inclination, intensity_distributionInclination, "sample cloud_mapInclination");

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_azimuth(new pcl::PointCloud<pcl::PointXYZI>);
        colorMapAzimuth(cloud_azimuth);
        pcl::visualization::PCLVisualizer vizA("PCL Azimuth Map");
        vizA.setBackgroundColor(1.0f, 1.0f, 1.0f);
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distributionAzimuth(cloud_azimuth, "intensity");
        vizA.addPointCloud<pcl::PointXYZI>(cloud_azimuth, intensity_distributionAzimuth, "sample cloud_mapAzimuth");
    }

    pcl::visualization::PCLVisualizer vizDepth("PCL Depth Map");
    vizDepth.setBackgroundColor(1.0f, 1.0f, 1.0f);
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distributionDepth(m_cloud_depth, "intensity");
    vizDepth.addPointCloud<pcl::PointXYZI>(m_cloud_depth, intensity_distributionDepth, "cloud_depth");
    vizDepth.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_depth");
    //viz2.addPlane(*coefficients, "plane");

    //pcl::visualization::PCLVisualizer vizConvexity("PCL Convexity Map");
    //vizConvexity.setBackgroundColor(1.0f, 1.0f, 1.0f);
    //pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distribution(m_cloud_convexity, "intensity");
    //vizConvexity.addPointCloud<pcl::PointXYZI>(m_cloud_convexity, intensity_distribution, "cloud_convexity");
    //vizConvexity.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud_convexity");
    //viz.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud_small, normals_small, 1, 0.01);

    pcl::visualization::PCLVisualizer vizCombined("PCL Combined Map");
    vizCombined.setBackgroundColor(1.0f, 1.0f, 1.0f);
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distributionCombinede(m_cloud_combined, "intensity");
    vizCombined.addPointCloud<pcl::PointXYZI>(m_cloud_combined, intensity_distributionCombinede, "m_cloud_combined");
    vizCombined.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "m_cloud_combined");
    //viz.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud_small, normals_small, 1, 0.01);

    while (!vizC.wasStopped())
    {
        vizC.spinOnce();
    }
}

void EntropyFilter::downsample(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in, float leaf_size,
                               pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_out)
{
    pcl::VoxelGrid<pcl::PointXYZRGB> vox_grid;
    vox_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
    vox_grid.setInputCloud(cloud_in);
    vox_grid.filter(*cloud_out);
}

void EntropyFilter::computePolyFitting(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, pcl::PointCloud<pcl::PointNormal> &mls_points)
{
    // Output has the PointNormal type in order to store the normals calculated by MLS

    // Create a KD-Tree
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);

    // Init object (second point type is for the normals, even if unused)
    pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointNormal> mls;

    mls.setComputeNormals(true);

    // Set parameters
    mls.setNumberOfThreads(4);
    mls.setInputCloud(cloud);
    mls.setPolynomialOrder(5);
    mls.setSearchMethod(tree);
    mls.setSearchRadius(0.03);

    // Reconstruct
    mls.process(mls_points);
    std::cout << "mls_points: " << mls_points.height << ", " << mls_points.width << std::endl;
}

void EntropyFilter::divideCloudNormals(pcl::PointCloud<pcl::PointNormal> &input, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                       pcl::PointCloud<pcl::Normal>::Ptr &normals)
{
    cloud->height = input.height;
    cloud->width = input.width;
    cloud->resize(input.size());
    normals->height = input.height;
    normals->width = input.width;
    normals->resize(input.size());
    for (int i = 0; i < input.size(); i++)
    {
        cloud->points[i].x = input.points[i].x;
        cloud->points[i].y = input.points[i].y;
        cloud->points[i].z = input.points[i].z;

        normals->points[i].normal_x = input.points[i].normal_x;
        normals->points[i].normal_y = input.points[i].normal_y;
        normals->points[i].normal_z = input.points[i].normal_z;
        normals->points[i].curvature = input.points[i].curvature;
    }
}

void EntropyFilter::getSpherical(pcl::PointCloud<pcl::Normal>::Ptr &cloud_normals, pcl::PointCloud<Spherical>::Ptr &spherical)
{
    spherical->width = cloud_normals->width;
    spherical->height = cloud_normals->height;
    spherical->resize(spherical->width * spherical->height);

    pcl::Normal data;
    for (size_t i = 0; i < cloud_normals->points.size(); ++i)
    {
        data = cloud_normals->points[i];
        spherical->points[i].azimuth = atan2(data.normal_z, data.normal_y);
        spherical->points[i].inclination = atan2(sqrt(data.normal_y * data.normal_y + data.normal_z * data.normal_z), data.normal_x);
    }
}

void EntropyFilter::normalizeEntropy(pcl::PointCloud<Spherical>::Ptr &spherical)
{

    float max_entropy = 0;
    for (int i = 0; i < spherical->size(); i++)
    {
        if (spherical->points[i].entropy > max_entropy)
            max_entropy = spherical->points[i].entropy;
    }

    for (int i = 0; i < spherical->size(); i++)
    {
        spherical->points[i].entropy_normalized = spherical->points[i].entropy / max_entropy;
    }
    std::cout << "max entropy : " << max_entropy << std::endl;
    _max_entropy = max_entropy;
}

//LOCAL HISTOGRAM and entropy calculation at the end.
//param[in]: point cloud normals in spherical coordinates
//param[in]: current point index in the cloud
//param[in]: vector of indeces of neighborhood points of considered on
void EntropyFilter::histogram2D(pcl::PointCloud<Spherical>::Ptr &spherical, int id0, std::vector<int> indices)
{
    int Hist[64][64] = {0};
    float step_inc = M_PI / 63;
    float step_az = (2 * M_PI) / 63;
    long bin_inc, bin_az;
    for (int i = 0; i < indices.size(); i++)
    {
        bin_inc = std::lroundf(spherical->points[indices[i]].inclination / step_inc);
        bin_az = std::lroundf((spherical->points[indices[i]].azimuth + M_PI) / step_az);
        if (bin_az < 0)
            std::cout << "erorr bin_az negative" << std::endl;

        Hist[bin_inc][bin_az]++;
    }
    bin_inc = std::lroundf(spherical->points[id0].inclination / step_inc);
    bin_az = std::lroundf((spherical->points[id0].azimuth + M_PI) / step_az);
    if (bin_az < 0)
        std::cout << "erorr bin_az negative" << std::endl;

    Hist[bin_inc][bin_az]++;

    float HistNorm[64][64] = {0};
    float max_value = 0;
    for (int i = 0; i < 64; i++)
    {
        for (int j = 0; j < 64; j++)
        {
            if (Hist[i][j] > max_value)
                max_value = Hist[i][j];
        }
    }

    for (int i = 0; i < 64; i++)
        for (int j = 0; j < 64; j++)
            HistNorm[i][j] = Hist[i][j] / max_value;

    //entropy calculation
    float entropy_value = 0;
    float temp_entropy = 0;
    for (int i = 0; i < 64; i++)
    {
        for (int j = 0; j < 64; j++)
        {
            temp_entropy = HistNorm[i][j] * log2(HistNorm[i][j]);
            if (!std::isnan(temp_entropy))
                entropy_value += temp_entropy;
        }
    }

    spherical->points[id0].entropy = -(entropy_value);
    //std::cout << "entropy value: " << spherical->points[id0].entropy << std::endl;
}

// LOCAL SEARCH
void EntropyFilter::local_search(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<Spherical>::Ptr &spherical,
                                 pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_combined)
{
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);
    pcl::PointXYZ searchPoint;

    for (int it = 0; it < cloud->points.size(); it++)
    {
        if (cloud_combined->points[it].intensity > 0)
        {
            searchPoint.x = cloud->points[it].x;
            searchPoint.y = cloud->points[it].y;
            searchPoint.z = cloud->points[it].z;

            // K nearest neighbor search
            std::vector<int> pointIdxNKNSearch(m_KNN);
            std::vector<float> pointNKNSquaredDistance(m_KNN);

            if (kdtree.nearestKSearch(searchPoint, m_KNN, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
                histogram2D(spherical, it, pointIdxNKNSearch);
            }
        }
        else
        {
            spherical->points[it].entropy = 0;
        }
    }
}

void EntropyFilter::segmentCloudEntropy(pcl::PointCloud<pcl::PointNormal> &cloud, pcl::PointCloud<Spherical>::Ptr &spherical,
                                        pcl::PointCloud<pcl::PointXYZ>::Ptr &output, float thresholdEntropy)
{
    pcl::PointXYZ p;
    for (int i = 0; i < spherical->size(); i++)
    {
        if (spherical->points[i].entropy_normalized > thresholdEntropy)
        {
            p.x = cloud.points[i].x;
            p.y = cloud.points[i].y;
            p.z = cloud.points[i].z;
            output->points.push_back(p);
        }
    }
    std::cout << "cloud segmented size: " << output->size() << std::endl;
}

void EntropyFilter::connectedComponets(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                       std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &cloud_clusters)
{
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.01); // 2cm
    ec.setMinClusterSize(50);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
            cloud_cluster->points.push_back(cloud->points[*pit]); //*

        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size() << " data points." << std::endl;
        cloud_clusters.push_back(cloud_cluster);
    }

    std::cout << "number of clusters found: " << cluster_indices.size() << std::endl;
}

void EntropyFilter::splitPointNormal(pcl::PointCloud<pcl::PointNormal>::Ptr &input, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                     pcl::PointCloud<pcl::Normal>::Ptr &normals)
{
    cloud->height = input->height;
    cloud->width = input->width;
    cloud->resize(input->size());
    normals->height = input->height;
    normals->width = input->width;
    normals->resize(input->size());
    for (int i = 0; i < input->size(); i++)
    {
        cloud->points[i].x = input->points[i].x;
        cloud->points[i].y = input->points[i].y;
        cloud->points[i].z = input->points[i].z;

        normals->points[i].normal_x = input->points[i].normal_x;
        normals->points[i].normal_y = input->points[i].normal_y;
        normals->points[i].normal_z = input->points[i].normal_z;
        normals->points[i].curvature = input->points[i].curvature;
    }
}

void EntropyFilter::computeDepthMap(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                    pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_out,
                                    pcl::ModelCoefficients &line)
{
    cloud_out->width = cloud->width;
    cloud_out->height = cloud->height;
    cloud_out->resize(cloud_out->width * cloud_out->height);

    Eigen::Vector4f line_pt, line_dir;
    double sqr_norm, value;

    std::vector<std::vector<int>> idx(4);
    line_pt[0] = line.values[0];
    line_pt[1] = line.values[1];
    line_pt[2] = line.values[2];
    line_pt[3] = 0;

    line_dir[0] = line.values[3];
    line_dir[1] = line.values[4];
    line_dir[2] = line.values[5];
    line_dir[3] = 0;

    sqr_norm = sqrt(line_dir.norm());

    for (int k = 0; k < cloud->size(); k++)
    {
        value = pcl::sqrPointToLineDistance(cloud->points[k].getVector4fMap(), line_pt, line_dir, sqr_norm);

        cloud_out->points[k].x = cloud->points[k].x;
        cloud_out->points[k].y = cloud->points[k].y;
        cloud_out->points[k].z = cloud->points[k].z;

        cloud_out->points[k].intensity = sqrt(value);
    }
}

void EntropyFilter::combineDepthAndCurvatureInfo(pcl::PointCloud<pcl::PointXYZI>::Ptr &depth,
                                                 pcl::PointCloud<pcl::Normal>::Ptr &normals,
                                                 pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_map)
{
    std::cout << depth->size() << std::endl;
    std::cout << normals->size() << std::endl;

    cloud_map->width = depth->width;
    cloud_map->height = depth->height;
    cloud_map->resize(depth->width * depth->height);

    std::vector<bool> result;
    result.resize(normals->size());
    for (int i = 0; i < normals->size(); i++)
    {
        if (normals->points[i].curvature >= m_curvature_threshold && depth->points[i].intensity <= m_depth_threshold)
        {
            cloud_map->points[i].intensity = 1;
        }
        else
        {
            cloud_map->points[i].intensity = 0;
        }

        cloud_map->points[i].x = depth->points[i].x;
        cloud_map->points[i].y = depth->points[i].y;
        cloud_map->points[i].z = depth->points[i].z;
    }
}