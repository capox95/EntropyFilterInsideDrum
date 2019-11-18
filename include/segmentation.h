#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/visualization/cloud_viewer.h>

/*
____________________________________________

sample1.pcd <---> sample13.pdc 
orientation: 0.9075 rad, 52 deg
line1: 0, 0.09, 0.11, 0, -0.787967, 0.615718;
line2: 0, 0.09, 0.11, 0, 0.615718, 0.787967;

data1.pcd <---> data4.pcd
orientation: 0.7592 rad, 43.5 deg
line1: 0, 0.04, 0.06, 0, -0.714004, 0.700142;
line2: 0, 0.04, 0.06, 0, 0.700142, 0.714004;
_____________________________________________

*/

class SegFilter
{
private:
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_, cloud_output_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_bw_, hull_vertices_, hull_result_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud_;

    pcl::ModelCoefficients line1_, line2_;
    float radiusDrum_, distanceCenterDrum_, drumDepth_;

    pcl::PointXYZ centerPoint_, segPoint1_, segPoint2_;
    std::vector<pcl::PointXYZ> pointsHull1_, pointsHull2_;

public:
    SegFilter() : source_(new pcl::PointCloud<pcl::PointXYZRGB>),
                  cloud_output_(new pcl::PointCloud<pcl::PointXYZRGB>),
                  transformed_cloud_(new pcl::PointCloud<pcl::PointXYZRGB>),
                  hull_vertices_(new pcl::PointCloud<pcl::PointXYZ>),
                  hull_result_(new pcl::PointCloud<pcl::PointXYZ>),
                  source_bw_(new pcl::PointCloud<pcl::PointXYZ>)

    {
        radiusDrum_ = 0.25;
        distanceCenterDrum_ = 0.4;
        drumDepth_ = 0.28;
    }

    void setSourceCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);

    void getOutputCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);

    void getDrumAxis(pcl::ModelCoefficients &axis);

    void compute();

    void visualizeSeg(bool flagSpin = false);

private:
    void transformation(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out);

    void computeDrumAxes(pcl::ModelCoefficients &line1, pcl::ModelCoefficients &line2);

    std::vector<pcl::PointXYZ> calculateHullPoints(pcl::PointXYZ &point1, Eigen::Vector3f &axis,
                                                   Eigen::Vector3f &vector, float radius_cylinder);

    void combineHullPoints(std::vector<pcl::PointXYZ> &p1, std::vector<pcl::PointXYZ> &p2,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

    void convexHullCrop(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_bw,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_vertices,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr &hull_result);

    void refineCrop(pcl::PointCloud<pcl::PointXYZ>::Ptr &hull_result, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out,
                    pcl::PointXYZ &point, Eigen::Vector3f &axis);
};