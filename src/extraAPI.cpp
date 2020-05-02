#include <Python.h>
#include "openmc/bank.h"
#include "openmc/capi.h"
#include "openmc/cmfd_solver.h"
#include "openmc/constants.h"
#include "openmc/cross_sections.h"
#include "openmc/dagmc.h"
#include "openmc/eigenvalue.h"
#include "openmc/error.h"
#include "openmc/geometry.h"
#include "openmc/geometry_aux.h"
#include "openmc/hdf5_interface.h"
#include "openmc/material.h"
#include "openmc/mesh.h"
#include "openmc/message_passing.h"
#include "openmc/mgxs_interface.h"
#include "openmc/nuclide.h"
#include "openmc/output.h"
#include "openmc/photon.h"
#include "openmc/plot.h"
#include "openmc/random_lcg.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/source.h"
#include "openmc/string_utils.h"
#include "openmc/summary.h"
#include "openmc/surface.h"
#include "openmc/thermal.h"
#include "openmc/timer.h"
#include "openmc/tallies/tally.h"
#include "openmc/volume_calc.h"

#include <cstddef>
#include <cstdlib> // for getenv
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Instructions for including in openmc:
// Before compiling openmc,
// Add this file to openmc/src/
// Add "void refresh_xml()" to openmc/include/capi.h
// Add "src/extraAPI.cpp" to openmc/CMakeLists.txt in list(APPEND libopenmc_SOURCES

// Add to openmc/CMakeLists.txt at line 335ish:
// find_package(PythonLibs REQUIRED)
// include_directories(${PYTHON_INCLUDE_DIRS})
// target_link_libraries(libopenmc ${PYTHON_LIBRARIES})

// Add refresh.py to openmc/openmc/lib/
// Add "from .refresh import *" to openmc/openmc/lib/__init__.py

using namespace openmc;
extern "C" void refresh_xml()
{
  // Clear results
  openmc_reset();

  // Reset timers
  reset_timers();

  // Reset global variables
  //settings::assume_separate = false;
  //settings::check_overlaps = false;
  //settings::confidence_intervals = false;
  settings::create_fission_neutrons = true;
  //settings::electron_treatment = ELECTRON_LED;
  //settings::energy_cutoff = {0.0, 1000.0, 0.0, 0.0};
  //settings::entropy_on = false;
  //settings::gen_per_batch = 1;
  settings::legendre_to_tabular = true;
  settings::legendre_to_tabular_points = -1;
  settings::n_particles = -1;
  //settings::output_summary = true;
  //settings::output_tallies = true;
  settings::particle_restart_run = false;
  settings::photon_transport = false;
  //settings::reduce_tallies = true;
  //settings::res_scat_on = false;
  //settings::res_scat_method = ResScatMethod::rvs;
  //settings::res_scat_energy_min = 0.01;
  //settings::res_scat_energy_max = 1000.0;
  settings::restart_run = false;
  settings::run_CE = true;
  settings::run_mode = -1;
  //settings::dagmc = false;
  //settings::source_latest = false;
  //settings::source_separate = false;
  settings::source_write = true;
  //settings::survival_biasing = false;
  ////settings::temperature_default = 293.6;
  ////settings::temperature_method = TEMPERATURE_NEAREST;
  ////settings::temperature_multipole = false;
  ////settings::temperature_range = {0.0, 0.0};
  ////settings::temperature_tolerance = 10.0;
  //settings::trigger_on = false;
  //settings::trigger_predict = false;
  //settings::trigger_batch_interval = 1;
  //settings::ufs_on = false;
  //settings::urr_ptables_on = true;
  //settings::verbosity = 7;
  //settings::weight_cutoff = 0.25;
  //settings::weight_survive = 1.0;
  settings::write_all_tracks = false;
  settings::write_initial_source = false;
  simulation::keff = 1.0;
  simulation::n_lost_particles = 0;
  //simulation::satisfy_triggers = false;
  simulation::total_gen = 0;
  simulation::entropy_mesh = nullptr;
  simulation::ufs_mesh = nullptr;
  model::root_universe = -1;
  openmc::openmc_set_seed(DEFAULT_SEED);


  free_memory_geometry();
  free_memory_surfaces();
  free_memory_material();
  free_memory_volume();
  free_memory_simulation();
  free_memory_settings();
  free_memory_source();
  free_memory_mesh();
  free_memory_tally();
  free_memory_bank();
  free_memory_cmfd();

#ifdef _OPENMP
  // If OMP_SCHEDULE is not set, default to a static schedule
  char* envvar = std::getenv("OMP_SCHEDULE");
  if (!envvar) {
    omp_set_schedule(omp_sched_static, 0);
  }
#endif

  // Initialize random number generator -- if the user specifies a seed, it
  // will be re-initialized later
  openmc_reset();
  openmc::openmc_set_seed(DEFAULT_SEED);
  openmc::read_settings_xml();
  openmc::read_cross_sections_xml();
  openmc::read_materials_xml();
  openmc::read_geometry_xml();
  //openmc_reset();

  double_2dvec nuc_temps(data::nuclide_map.size());
  double_2dvec thermal_temps(data::thermal_scatt_map.size());
  finalize_geometry(nuc_temps, thermal_temps);

  if (settings::run_mode != RUN_MODE_PLOTTING) {
    simulation::time_read_xs.start();
    if (settings::run_CE) {
      // Read continuous-energy cross sections
      //read_ce_cross_sections(nuc_temps, thermal_temps);
    } else {
      // Create material macroscopic data for MGXS
      read_mgxs();
      create_macro_xs();
    }
    simulation::time_read_xs.stop();
  }
  for (auto& mat : model::materials) {
    mat->finalize();
  }
  read_tallies_xml();

  // Initialize distribcell_filters
  prepare_distribcell();

  if (settings::run_mode == RUN_MODE_PLOTTING) {
    // Read plots.xml if it exists
    read_plots_xml();
    if (mpi::master && settings::verbosity >= 5) print_plot();

  } else {
    // Write summary information
    if (mpi::master && settings::output_summary) write_summary();

    // Warn if overlap checking is on
    if (mpi::master && settings::check_overlaps) {
      warning("Cell overlap checking is ON.");
    }
  }
  if (settings::particle_restart_run) settings::run_mode = RUN_MODE_PARTICLE;
}

extern "C" PyObject* openmc_run_with_errorcheck()
{
  try{
    int err = openmc_run();
    return PyLong_FromLong(err);
  }
  catch(...){
    PyErr_SetObject(PyExc_SystemError, PyBytes_FromString("Openmc Crash"));
    return NULL;
  }
}

