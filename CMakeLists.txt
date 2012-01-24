project(interactive_roots)

cmake_minimum_required(VERSION 2.8)

find_package(Qt4 4.6.2 COMPONENTS QtCore QtGui)

include_directories(${QT_INCLUDES})

qt4_wrap_cpp(moc_srcs
	interactive_roots.h
)
set(srcs
	${moc_srcs}
	interactive_roots.cpp
)

set(CMAKE_BUILD_TYPE Release CACHE STRING
	"Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel."
	FORCE)
set(CMAKE_CXX_FLAGS "-Wall -fopenmp")

add_executable(interactive_roots ${srcs})
target_link_libraries(interactive_roots ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY})


