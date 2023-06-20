// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>
#include <aliceVision/sfm/pipeline/regionsIO.hpp>
#include <aliceVision/voctree/Database.hpp>
#include <aliceVision/voctree/databaseIO.hpp>
#include <aliceVision/voctree/VocabularyTree.hpp>
#include <aliceVision/voctree/descriptorLoader.hpp>
#include <aliceVision/matching/IndMatch.hpp>
#include <aliceVision/system/Logger.hpp>
#include <aliceVision/cmdline/cmdline.hpp>
#include <aliceVision/system/main.hpp>
#include <aliceVision/types.hpp>
#include <aliceVision/utils/convert.hpp>

#include <Eigen/Core>

#include <boost/program_options.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/tail.hpp>

#include <iostream>
#include <fstream>
#include <ostream>
#include <string>
#include <chrono>
#include <iomanip>

// These constants define the current software version.
// They must be updated when the command line is changed.
#define ALICEVISION_SOFTWARE_VERSION_MAJOR 1
#define ALICEVISION_SOFTWARE_VERSION_MINOR 0

static const int DIMENSION = 128;

using namespace boost::accumulators;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace aliceVision;
using namespace aliceVision::feature;

typedef aliceVision::feature::Descriptor<float, DIMENSION> DescriptorFloat;
typedef aliceVision::feature::Descriptor<unsigned char, DIMENSION> DescriptorUChar;

bool saveSparseHistogramPerImage(const std::string& filename, const aliceVision::voctree::SparseHistogramPerImage& docs)
{
    std::ofstream fileout(filename);
    if(!fileout.is_open())
        return false;

    for(const auto& d : docs)
    {
        fileout << "d{" << d.first << "} = [";
        for(const auto& i : d.second)
            fileout << i.first << ", ";
        fileout << "]\n";
    }

    fileout.close();
    return true;
}

static const std::string programDescription =
    "This program is used to create a database with a provided dataset of image descriptors using a trained vocabulary "
    "tree.\n "
    "The database is then queried optionally with another set of images in order to retrieve for each image the set of "
    "most similar images in the dataset\n"
    "If another set of images is not provided, the program will perform a sanity check of the database by querying the "
    "database using the same images used to build it\n"
    "It takes as input either a list.txt file containing the a simple list of images (bundler format and older "
    "AliceVision version format)\n"
    "or a sfm_data file (JSON) containing the list of images. In both cases it is assumed that the .desc to load are "
    "in the same folder as the input file\n"
    "For the vocabulary tree, it takes as input the input.tree (and the input.weight) file generated by createVoctree\n"
    "As a further output option (--outdir), it is possible to specify a folder in which it will create, for each query "
    "image (be it a query image of querylist or an image of keylist)\n"
    "it creates a folder with the same name of the image, inside which it creates a list of symbolic links to all the "
    "similar images found. The symbolic link naming convention\n"
    "is matchNumber.filename, where matchNumber is the relevant position of the image in the list of matches ([0-r]) "
    "and filename is its image file (eg image.jpg)\n";

/*
 * This program is used to create a database with a provided dataset of image descriptors using a trained vocabulary
 * tree The database is then queried with the same images in order to retrieve for each image the set of most similar
 * images in the dataset
 */
int aliceVision_main(int argc, char** argv)
{
    std::string verboseLevel = system::EVerboseLevel_enumToString(system::Logger::getDefaultVerboseLevel());
    /// the filename for the voctree weights
    std::string weightsName;
    /// flag for the optional weights file
    bool withWeights = false;
    /// the filename of the voctree
    std::string treeName;
    /// the file containing the list of features to use to build the database
    std::string sfmDataFilename;
    /// the file containing the list of features to use as query
    std::string querySfmDataFilename;
    std::vector<std::string> featuresFolders;
    /// the file in which to save the results
    std::string outfile;
    /// the folder in which save the symlinks of the similar images
    std::string outDir;
    /// the file where to save the document map in matlab format
    std::string documentMapFile;
    std::string describerMethod = feature::EImageDescriberType_enumToString(feature::EImageDescriberType::SIFT);
    /// flag for the optional output file
    bool withOutput = false;
    /// flag for the optional output folder to save the symlink of the similar images
    bool withOutDir = false;
    /// flag for the optional path to the SfMData file to use for querying the database
    bool withQuery = false;
    /// it produces an output readable by matlab
    bool matlabOutput = false;
    /// the number of matches to retrieve for each image
    std::size_t numImageQuery = 10;
    std::string distance;
    int Nmax = 0;

    aliceVision::sfmData::SfMData sfmData;
    aliceVision::sfmData::SfMData* querySfmData;

    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()
        ("input,i", po::value<std::string>(&sfmDataFilename)->required(),
         "A SfMData file.")
        ("tree,t", po::value<std::string>(&treeName)->required(),
         "Input name for the tree file.")
        ("featuresFolders,f", po::value<std::vector<std::string>>(&featuresFolders)->multitoken()->required(),
         "Path to folder(s) containing the extracted features.");

    po::options_description optionalParams("Optional parameters");
    optionalParams.add_options()
        ("weights,w", po::value<std::string>(&(weightsName)),
         "Input name for the weight file, if not provided the weights will be computed on the "
         "database built with the provided set.")
        ("querySfmDataFilename,q", po::value<std::string>(&querySfmDataFilename),
         "Path to the SfMData file to be used for querying the database.")
        ("saveDocumentMap", po::value<std::string>(&documentMapFile),
         "A Matlab file .m where to save the document map of the created database.")
        ("outdir", po::value<std::string>(&outDir),
         "Path to the folder in which save the symlinks of the similar images (it will be created if it does not "
         "exist).")
        ("describerMethod,m", po::value<std::string>(&describerMethod)->default_value(describerMethod),
         "Method to use to describe an image.")
        ("results,r", po::value<std::size_t>(&numImageQuery)->default_value(numImageQuery),
         "The number of matches to retrieve for each image, 0 to retrieve all the images.")
        ("matlab,", po::value<bool>(&matlabOutput)->default_value(matlabOutput),
         "It produces an output readable by Matlab.")
        ("outfile,o", po::value<std::string>(&outfile),
         "Name of the output file.")
        ("Nmax,n", po::value<int>(&Nmax)->default_value(Nmax),
         "Number of features extracted from the .feat files.")
        ("distance,d", po::value<std::string>(&distance)->default_value("strongCommonPoints"),
         "Distance used.");

    aliceVision::CmdLine cmdline(programDescription + "AliceVision voctreeQueryUtility");
    cmdline.add(requiredParams);
    cmdline.add(optionalParams);

    if(!cmdline.execute(argc, argv))
    {
        return EXIT_FAILURE;
    }

    if(!weightsName.empty())
    {
        withWeights = true;
    }
    if(!outfile.empty())
    {
        withOutput = true;
    }
    if(!querySfmDataFilename.empty())
    {
        withQuery = true;
    }
    if(!outDir.empty())
    {
        withOutDir = true;
    }

    // load vocabulary tree

    ALICEVISION_LOG_INFO("Loading vocabulary tree\n");
    aliceVision::voctree::VocabularyTree<DescriptorFloat> tree(treeName);
    ALICEVISION_LOG_INFO("tree loaded with\n\t" << tree.levels() << " levels\n\t" << tree.splits()
                                                << " branching factor");

    // create the database

    ALICEVISION_LOG_INFO("Creating the database...");
    // Add each object (document) to the database
    aliceVision::voctree::Database db(tree.words());

    if(withWeights)
    {
        ALICEVISION_LOG_INFO("Loading weights...");
        db.loadWeights(weightsName);
    }
    else
    {
        ALICEVISION_LOG_INFO("No weights specified, skipping...");
    }

    if(withOutDir)
    {
        // load the json for the dataset used to build the database
        if(sfmDataIO::Load(sfmData, sfmDataFilename, sfmDataIO::ESfMData(sfmDataIO::VIEWS | sfmDataIO::INTRINSICS)))
        {
            ALICEVISION_LOG_INFO("SfMData loaded from " << sfmDataFilename << " containing: ");
            ALICEVISION_LOG_INFO("\tnumber of views: " << sfmData.getViews().size());
        }
        else
        {
            ALICEVISION_LOG_ERROR("Could not load the SfMData file '" << sfmDataFilename << "'");
            return EXIT_FAILURE;
        }

        // load the json for the dataset used to query the database
        if(withQuery)
        {
            querySfmData = new aliceVision::sfmData::SfMData();
            if(sfmDataIO::Load(*querySfmData, querySfmDataFilename,
                               sfmDataIO::ESfMData(sfmDataIO::VIEWS | sfmDataIO::INTRINSICS)))
            {
                ALICEVISION_LOG_INFO("SfMData loaded from " << querySfmDataFilename << " containing: ");
                ALICEVISION_LOG_INFO("\tnumber of views: " << querySfmData->getViews().size());
            }
            else
            {
                ALICEVISION_LOG_ERROR("Could not load the SfMData file '" << querySfmDataFilename << "'");
                return EXIT_FAILURE;
            }
        }
        else
        {
            // otherwise sfmdataQuery is just a link to the dataset sfmdata
            querySfmData = &sfmData;
        }

        // create recursively the provided out dir
        if(!fs::exists(fs::path(outDir)))
        {
            // ALICEVISION_COUT("creating folder" << outDir);
            fs::create_directories(fs::path(outDir));
        }
    }

    // read the descriptors and populate the database

    ALICEVISION_LOG_INFO("Reading descriptors from " << sfmDataFilename);
    auto detect_start = std::chrono::steady_clock::now();
    std::size_t numTotFeatures =
        aliceVision::voctree::populateDatabase<DescriptorUChar>(sfmData, featuresFolders, tree, db, Nmax);
    auto detect_end = std::chrono::steady_clock::now();
    auto detect_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(detect_end - detect_start);

    if(numTotFeatures == 0)
    {
        ALICEVISION_LOG_ERROR("No descriptors loaded!!");
        return EXIT_FAILURE;
    }

    ALICEVISION_LOG_INFO("Done! " << db.getSparseHistogramPerImage().size()
                                  << " sets of descriptors read for a total of " << numTotFeatures << " features");
    ALICEVISION_LOG_INFO("Reading took " << detect_elapsed.count() << " sec");

    if(!documentMapFile.empty())
    {
        saveSparseHistogramPerImage(documentMapFile, db.getSparseHistogramPerImage());
    }

    if(!withWeights)
    {
        // If we don't have an input weight file, we compute weights based on the
        // current database.
        ALICEVISION_LOG_INFO("Computing weights...");
        db.computeTfIdfWeights();
    }

    // query documents or sanity check

    std::map<std::size_t, voctree::DocMatches> allDocMatches;
    std::size_t wrong = 0;
    if(numImageQuery == 0)
    {
        // if 0 retrieve the score for all the documents of the database
        numImageQuery = db.size();
    }
    std::ofstream fileout;
    if(withOutput)
    {
        fileout.open(outfile, std::ofstream::out);
    }

    std::map<std::size_t, voctree::SparseHistogram> histograms;

    // if the query list is not provided
    if(!withQuery)
    {
        // do a sanity check
        ALICEVISION_LOG_INFO("Sanity check: querying the database with the same documents");
        db.sanityCheck(numImageQuery, allDocMatches);
    }
    else
    {
        // otherwise query the database with the provided query list
        ALICEVISION_LOG_INFO("Querying the database with the documents in " << querySfmDataFilename);
        voctree::queryDatabase<DescriptorUChar>(*querySfmData, featuresFolders, tree, db, numImageQuery, allDocMatches,
                                                histograms, distance, Nmax);
    }

    // Load the corresponding RegionsPerView
    // Get imageDescriberMethodType
    EImageDescriberType describerType = EImageDescriberType_stringToEnum(describerMethod);

    if((describerType != EImageDescriberType::SIFT) && (describerType != EImageDescriberType::SIFT_FLOAT))
    {
        ALICEVISION_LOG_ERROR("Invalid describer method." << std::endl);
        return EXIT_FAILURE;
    }

    feature::RegionsPerView regionsPerView;
    if(!aliceVision::sfm::loadRegionsPerView(regionsPerView, sfmData, featuresFolders, {describerType}))
    {
        ALICEVISION_LOG_ERROR("Invalid regions." << std::endl);
        return EXIT_FAILURE;
    }

    aliceVision::matching::PairwiseSimpleMatches allMatches;

    for(auto docMatches : allDocMatches)
    {
        const aliceVision::voctree::DocMatches& matches = docMatches.second;
        fs::path dirname;
        ALICEVISION_LOG_INFO("Camera: " << docMatches.first);
        ALICEVISION_LOG_INFO("query document " << docMatches.first << " has " << matches.size() << " matches\tBest "
                                               << matches[0].id << " with score " << matches[0].score);
        if(withOutput)
        {
            if(!matlabOutput)
            {
                fileout << "Camera: " << docMatches.first << std::endl;
            }
            else
            {
                fileout << "m{" << docMatches.first + 1 << "}=";
                fileout << matches;
            }
        }
        if(withOutDir)
        {
            // create a new folder inside outDir with the same name as the query image
            // the query image can be either from the dataset or from the query list if provided

            // to put a symlink to the query image too
            fs::path absoluteFilename; //< the abs path to the image
            fs::path sylinkName;       //< the name used for the symbolic link

            // get the dirname from the filename

            aliceVision::sfmData::Views::const_iterator it = querySfmData->getViews().find(docMatches.first);
            if(it == querySfmData->getViews().end())
            {
                // this is very wrong
                ALICEVISION_LOG_ERROR("Could not find the image file for the document " << docMatches.first << "!");
                return EXIT_FAILURE;
            }
            sylinkName = fs::path(it->second->getImagePath()).filename();
            dirname = fs::path(outDir) / sylinkName;
            absoluteFilename = it->second->getImagePath();
            fs::create_directories(dirname);
            fs::create_symlink(absoluteFilename, dirname / sylinkName);

            // Perform features matching
            const aliceVision::voctree::SparseHistogram& currentHistogram = histograms.at(docMatches.first);

            for(const auto comparedPicture : matches)
            {
                aliceVision::voctree::SparseHistogram comparedHistogram = histograms.at(comparedPicture.id);
                aliceVision::Pair indexImagePair = aliceVision::Pair(docMatches.first, comparedPicture.id);

                // Get the regions for the current view pair.
                // const aliceVision::feature::SIFT_Regions& lRegions =
                // dynamic_cast<aliceVision::feature::SIFT_Regions>(regionsPerView->getRegions(indexImagePair.first);
                // const aliceVision::feature::SIFT_Regions& rRegions =
                // dynamic_cast<aliceVision::feature::SIFT_Regions>(regionsPerView->getRegions(indexImagePair.second);

                // Distances Vector
                // const std::vector<float> distances;

                aliceVision::matching::IndMatches featureMatches;

                for(const auto& currentLeaf : currentHistogram)
                {
                    if((currentLeaf.second.size() == 1))
                    {
                        auto leafRightIt = comparedHistogram.find(currentLeaf.first);
                        if(leafRightIt == comparedHistogram.end())
                            continue;
                        if(leafRightIt->second.size() != 1)
                            continue;

                        const Regions& siftRegionsLeft = regionsPerView.getRegions(docMatches.first, describerType);
                        const Regions& siftRegionsRight = regionsPerView.getRegions(comparedPicture.id, describerType);

                        double dist = siftRegionsLeft.SquaredDescriptorDistance(
                            currentLeaf.second[0], &siftRegionsRight, leafRightIt->second[0]);
                        aliceVision::matching::IndMatch currentMatch =
                            aliceVision::matching::IndMatch(currentLeaf.second[0], leafRightIt->second[0]
#ifdef ALICEVISION_DEBUG_MATCHING
                                                            , dist
#endif
                            );
                        featureMatches.push_back(currentMatch);

                        // TODO: distance computation
                    }
                }

                allMatches[indexImagePair] = featureMatches;

                // TODO: display + symlinks
            }
        }

        // now parse all the returned matches
        for(std::size_t j = 0; j < matches.size(); ++j)
        {
            ALICEVISION_LOG_INFO("\t match " << matches[j].id << " with score " << matches[j].score);
            // ALICEVISION_CERR("" << i->first << " " << matches[j].id << " " << matches[j].score);
            if(withOutput && !matlabOutput)
                fileout << docMatches.first << " " << matches[j].id << " " << matches[j].score << std::endl;

            if(withOutDir)
            {
                // create a new symbolic link inside the current folder pointing to
                // the relevant matching image
                fs::path absoluteFilename; //< the abs path to the image
                fs::path sylinkName;       //< the name used for the symbolic link

                // get the dirname from the filename
                aliceVision::sfmData::Views::const_iterator it = sfmData.getViews().find(matches[j].id);
                if(it != sfmData.getViews().end())
                {
                    absoluteFilename = it->second->getImagePath();
                    sylinkName = fs::path(utils::toStringZeroPadded(j, 4) + "." + std::to_string(matches[j].score) +
                                          "." + absoluteFilename.filename().string());
                }
                else
                {
                    // this is very wrong
                    ALICEVISION_LOG_ERROR("Could not find the image file for the document " << matches[j].id << "!");
                    return EXIT_FAILURE;
                }
                fs::create_symlink(absoluteFilename, dirname / sylinkName);
            }
        }

        if(!withQuery)
        {
            // only for the sanity check, check if the best matching image is the document itself
            if(docMatches.first != matches[0].id)
            {
                ++wrong;
                ALICEVISION_LOG_INFO("##### wrong match for document " << docMatches.first);
            }
        }
    }

#ifdef ALICEVISION_DEBUG_MATCHING
    std::cout << " ---------------------------- \n" << std::endl;
    std::cout << "Matching distances - Histogram: \n" << std::endl;
    std::map<int, int> stats;
    for(const auto& imgMatches : allMatches)
    {
        if(imgMatches.first.first == imgMatches.first.second)
            // Ignore auto-match
            continue;

        for(const aliceVision::matching::IndMatch& featMatches : imgMatches.second)
        {
            int d = std::floor(featMatches._distance / 1000.0);
            if(stats.find(d) != stats.end())
                stats[d] += 1;
            else
                stats[d] = 1;
        }
    }
    for(const auto& stat : stats)
    {
        std::cout << stat.first << "\t" << stat.second << std::endl;
    }
#endif

    if(!withQuery)
    {
        if(wrong)
            ALICEVISION_LOG_INFO("there are " << wrong << " wrong matches");
        else
            ALICEVISION_LOG_INFO("no wrong matches!");
    }

    if(withOutput)
    {
        fileout.close();
    }

    return EXIT_SUCCESS;
}
