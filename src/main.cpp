#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>

#include <chrono>
#include <ctime>

#include "../include/entropy.h"
#include "../include/pointpose.h"

void cloudFiltering(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_filtered)
{
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-0.2, 0.2);
    //pass.setFilterLimitsNegative (true);
    pass.filter(*cloud_filtered);

    pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
    // build the filter
    outrem.setInputCloud(cloud_filtered);
    outrem.setRadiusSearch(0.01);
    outrem.setMinNeighborsInRadius(20);
    // apply filter
    outrem.filter(*cloud_filtered);

    /*
    pcl::PassThrough<pcl::PointXYZRGB> pass2;
    pass2.setInputCloud(cloud_filtered);
    pass2.setFilterFieldName("z");
    pass2.setFilterLimits(-0.1, 0.5);
    //pass.setFilterLimitsNegative (true);
    pass2.filter(*cloud_filtered);
    */
}

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

    cloudFiltering(source, source_filtered);

    //------------------------------------------
    Eigen::Vector3f basketCenter;
    basketCenter.x() = 0.0;
    basketCenter.y() = 0.0;
    basketCenter.z() = 0.0;

    Eigen::Vector3f basketAxisDir;
    basketAxisDir.x() = 0.0205354;
    basketAxisDir.y() = -0.631626;
    basketAxisDir.z() = 0.521389;

    pcl::ModelCoefficients line;
    std::vector<float> values = {basketCenter.x(), basketCenter.y(), basketCenter.z(),
                                 basketAxisDir.x(), basketAxisDir.y(), basketAxisDir.z()};
    line.values = values;

    pcl::visualization::PCLVisualizer viz("PCL filtering");
    viz.addCoordinateSystem(0.2, "coord");
    viz.setBackgroundColor(0.0, 0.0, 0.5);
    viz.addPointCloud(source_filtered, "source_filtered");
    viz.addLine(line, "line");

    auto startE = std::chrono::steady_clock::now();
    // ENTROPY FILTER -----------------------------------------------------------------------
    //
    EntropyFilter ef;
    ef.setInputCloud(source_filtered);
    ef.setDownsampleLeafSize(0.0001); // size of the leaf for downsampling the cloud, value in meters. Default = 5 mm
    ef.setEntropyThreshold(0.5);      // Segmentation performed for all points with normalized entropy value above this
    ef.setKLocalSearch(500);          // Nearest Neighbour Local Search
    ef.setCurvatureThreshold(0.05);   // Curvature Threshold for the computation of Entropy
    ef.setDepthThreshold(0.25);       //0.29         // if the segment region has a value of depth lower than this -> not graspable (value in meters)
    ef.setDrumAxis(line);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_result(new pcl::PointCloud<pcl::PointXYZ>);
    bool entropy_result = ef.compute(cloud_result);
    if (entropy_result == false)
        return -1;

    // GRASP POINT --------------------------------------------------------------------------
    PointPose pp;
    pp.setSourceCloud(source);
    pp.setInputCloud(cloud_result);
    Eigen::Affine3d transformation;
    pp.computeGraspPoint(transformation);

    //time computation
    auto endE = std::chrono::steady_clock::now();
    auto diff2 = endE - startE;
    std::cout << "duration entropy filter: " << std::chrono::duration<double, std::milli>(diff2).count() << " ms" << std::endl;

    pp.visualizeGrasp();
    ef.visualizeAll(false);

    return 0;
}
