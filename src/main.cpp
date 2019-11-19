#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>

#include <chrono>
#include <ctime>

#include "../include/entropy.h"
#include "../include/pointpose.h"
#include "../include/segmentation.h"

//----------------------------------------------------------------------------- //
int main(int argc, char **argv)
{

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(argv[1], *source) == -1)
    {
        PCL_ERROR(" error opening file ");
        return (-1);
    }
    std::cout << "cloud orginal size: " << source->size() << std::endl;

    SegFilter sf;
    sf.setSourceCloud(source);
    sf.compute();
    sf.visualizeSeg();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr segfilter_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    sf.getOutputCloud(segfilter_cloud);

    pcl::ModelCoefficients axis;
    sf.getDrumAxis(axis);

    // ENTROPY FILTER -----------------------------------------------------------------------
    //
    EntropyFilter ef;
    ef.setInputCloud(segfilter_cloud);
    ef.setDownsampleLeafSize(0.0001); // size of the leaf for downsampling the cloud, value in meters. Default = 5 mm
    ef.setEntropyThreshold(0.75);      // Segmentation performed for all points with normalized entropy value above this
    ef.setKLocalSearch(500);          // Nearest Neighbour Local Search
    ef.setCurvatureThreshold(0.03);   // Curvature Threshold for the computation of Entropy
    ef.setDepthThreshold(0.23);       //0.29         // if the segment region has a value of depth lower than this -> not graspable (value in meters)
    ef.setDrumAxis(axis);

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds_result;
    bool entropy_result = ef.compute(clouds_result);
    if (entropy_result == false)
        return -1;

    // GRASP POINT --------------------------------------------------------------------------

    PointPose pp;
    pp.setSourceCloud(source);
    pp.setInputVectorClouds(clouds_result);
    pp.setDrumAxis(axis);

    std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>> transformation_vector;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points_vector;
    int number = pp.compute(points_vector, transformation_vector);

    std::cout << "---------------------------------------------- " << std::endl;
    for (int i = 0; i < number; i++)
    {
        std::cout << "Transformation Matrix: \n"
                  << transformation_vector.at(i).matrix() << std::endl;
        std::cout << std::endl;
        std::cout << "Point on Drum Axis: \n"
                  << points_vector.at(i) << std::endl;
        std::cout << "---------------------------------------------- " << std::endl;
    }

    //-------------------------------------------------------------------------------------
    pp.visualizeGrasp();
    ef.visualizeAll(false);

    return 0;
}
