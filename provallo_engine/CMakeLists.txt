cmake_minimum_required(VERSION 3.0)

#set the project 
project(provallo_engine-${_system_name}_${_host_id})    

#set the version
set(VERSION_MAJOR 0)
set(VERSION_MINOR 1)
set(VERSION_PATCH 0)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#set the build type
set(CMAKE_BUILD_TYPE Debug)

#set the compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Wpedantic -Werror")

#set the include directories
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/util ${CMAKE_CURRENT_SOURCE_DIR}/data ${CMAKE_CURRENT_SOURCE_DIR}/glue ${CMAKE_CURRENT_SOURCE_DIR}/decision_engine  ${CMAKE_CURRENT_SOURCE_DIR}/parsers ${CMAKE_CURRENT_SOURCE_DIR}/helpers ${CMAKE_CURRENT_SOURCE_DIR}/statistics ${CMAKE_CURRENT_SOURCE_DIR}/third_party )   

#enable AVX options to the compiler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2 -mfma -mfpmath=sse -march=native -mtune=native -O3 -fPIC -fno-omit-frame-pointer -fno-strict-aliasing  -fno-stack-protector -fno-strict-overflow -fno-delete-null-pointer-checks -fno-lifetime-dse -fno-merge-all-constants")    


#set the source files 


set (PROVALLO_ENGINE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/data/minidal.cpp ${CMAKE_CURRENT_SOURCE_DIR}/decision_engine/optimizers.cpp ${CMAKE_CURRENT_SOURCE_DIR}/decision_engine/split_utils.cpp 
${CMAKE_CURRENT_SOURCE_DIR}/decision_engine/matrix.cpp ${CMAKE_CURRENT_SOURCE_DIR}/decision_engine/classdist.cpp ${CMAKE_CURRENT_SOURCE_DIR}/decision_engine/attribute.cpp ${CMAKE_CURRENT_SOURCE_DIR}/decision_engine/kNearestNeighbor.cpp ${CMAKE_CURRENT_SOURCE_DIR}/decision_engine/decision_engine.cpp ${CMAKE_CURRENT_SOURCE_DIR}/decision_engine/distributedkmeans.cpp ${CMAKE_CURRENT_SOURCE_DIR}/decision_engine/parameters.cpp ${CMAKE_CURRENT_SOURCE_DIR}/decision_engine/pipelinebuilder.cpp ${CMAKE_CURRENT_SOURCE_DIR}/decision_engine/dataset.cpp  
${CMAKE_CURRENT_SOURCE_DIR}/decision_engine/neuralhelper.cpp ${CMAKE_CURRENT_SOURCE_DIR}/decision_engine/classifier.cpp ${CMAKE_CURRENT_SOURCE_DIR}/statistics/statcollector.cpp ${CMAKE_CURRENT_SOURCE_DIR}/glue/linuxglue.cpp ${CMAKE_CURRENT_SOURCE_DIR}/glue/turnserverglue.cpp ${CMAKE_CURRENT_SOURCE_DIR}/glue/gluelinuxinvoker.cpp ${CMAKE_CURRENT_SOURCE_DIR}/glue/glueprocessinfo.cpp ${CMAKE_CURRENT_SOURCE_DIR}/util/monitorpolicy.cpp ${CMAKE_CURRENT_SOURCE_DIR}/util/basic_encoding.cpp ${CMAKE_CURRENT_SOURCE_DIR}/util/portanalyzer.cpp ${CMAKE_CURRENT_SOURCE_DIR}/util/aes_wrapper.cpp ${CMAKE_CURRENT_SOURCE_DIR}/util/ip_address.cpp ${CMAKE_CURRENT_SOURCE_DIR}/util/procwatcher.cpp ${CMAKE_CURRENT_SOURCE_DIR}/util/hash_tools.cpp ${CMAKE_CURRENT_SOURCE_DIR}/util/interface_info.cpp ${CMAKE_CURRENT_SOURCE_DIR}/util/safehttp.cpp ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite.cpp ${CMAKE_CURRENT_SOURCE_DIR}/helpers/netlinkstatshelper.cpp ${CMAKE_CURRENT_SOURCE_DIR}/helpers/netlinkhelper.cpp ${CMAKE_CURRENT_SOURCE_DIR}/helpers/netfilterhelper.cpp ${CMAKE_CURRENT_SOURCE_DIR}/decision_engine/fast_matrix_forest.cpp )

#set the source files for the prefrontal test executable :
set(PFC_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/trainer_box/prefrontal_cortex_simulation.cpp ${PROVALLO_ENGINE_SOURCES} ) 
#add the test executable:
add_executable(pfc_test ${PFC_TEST_SOURCES} )
#set the source files
set(SOURCE_FILES ${PROVALLO_ENGINE_SOURCES}     
        ${CMAKE_CURRENT_SOURCE_DIR}/trainer_box/main.cpp)   

#set the executable 
add_executable(provallo_engine ${SOURCE_FILES})




#set the libraries
target_link_libraries(provallo_engine pthread crypto ssl elf nfnetlink netfilter_queue sqlite3 ) 

#set the libraries
target_link_libraries(pfc_test pthread crypto ssl elf nfnetlink netfilter_queue sqlite3 ) 
#set the install target
install(TARGETS provallo_engine pfc_test DESTINATION bin)

install(TARGETS  pfc_test DESTINATION bin)

#add compilation flags for the test
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11  -Wno-unused-local-typedefs   -Wall -Wextra -Wno-deprecated  -Werror -fprofile-arcs -ftest-coverage ")   

#set the cpack
set(CPACK_GENERATOR "DEB")
set(CPACK_PACKAGE_NAME "provallo_engine")
set(CPACK_PACKAGE_VERSION_MAJOR ${VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${VERSION_PATCH})
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Provallo Team")
set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "Provallo Engine")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libssl-dev, libcrypto++-dev, libelf-dev, libnetfilter-queue-dev, libnfnetlink-dev libsqlite3-dev")

include(CPack)
