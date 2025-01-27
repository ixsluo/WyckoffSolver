project(
    'wyckoff_solver',
    'c', 'cpp',
    version: '0.0.1',
    license: 'BSD-3',
    meson_version: '>=0.64.0',
    default_options: ['warning_level=2'],
)

py_mod = import('python')
py3 = py_mod.find_installation(pure: false)
py_dep = py3.dependency()

incdir_numpy = run_command(py3,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py3,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)


cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

# Adding at project level causes many spurious -lgfortran flags.
add_languages('fortran', native: false)
ff = meson.get_compiler('fortran')
if ff.get_id() == 'gcc'
  # -std=legacy is not supported by all Fortran compilers, but very useful with
  # gfortran since it avoids a ton of warnings that we don't care about.
  # Needs fixing in Meson, see https://github.com/mesonbuild/meson/issues/11633.
  add_project_arguments('-std=legacy', language: 'fortran')
endif

if ff.has_argument('-Wno-conversion')
  add_project_arguments('-Wno-conversion', language: 'fortran')
endif

if host_machine.system() == 'darwin'
  if cc.has_link_argument('-Wl,-dead_strip')
    # Allow linker to strip unused symbols
    add_project_link_arguments('-Wl,-dead_strip', language : ['c', 'cpp', 'fortran'])
  endif
endif

# Intel compilers default to fast-math, so disable it if we detect Intel
# compilers. A word of warning: this may not work with the conda-forge
# compilers, because those have the annoying habit of including lots of flags
# that are gcc-specific in CFLAGS/CXXFLAGS/FFLAGS, which throws off the
# detection logic below. You have to remove the wrong flags (only `-isystem`
# is actually needed, everything else shouldn't be there).
_intel_cflags = []
_intel_fflags = []
if cc.get_id() in ['intel', 'intel-llvm']
  _intel_cflags += cc.get_supported_arguments('-fp-model=strict')
elif cc.get_id() in ['intel-cl', 'intel-llvm-cl']
  _intel_cflags += cc.get_supported_arguments('/fp:strict')
endif
if ff.get_id() in ['intel', 'intel-llvm']
  _intel_fflags = ff.get_supported_arguments('-fp-model=strict')
  minus0_arg = ['-assume', 'minus0']
  if ff.has_multi_arguments(minus0_arg)
    _intel_fflags += minus0_arg
  endif
elif ff.get_id() in ['intel-cl', 'intel-llvm-cl']
  # Intel Fortran on Windows does things differently, so deal with that
  # (also specify dynamic linking and the right name mangling)
  _intel_fflags = ff.get_supported_arguments(
    '/fp:strict', '/MD', '/names:lowercase', '/assume:underscore',
    '/assume:minus0'
  )
endif
add_project_arguments(_intel_cflags, language: ['c', 'cpp'])
add_project_arguments(_intel_fflags, language: 'fortran')

subdir('wyckoff_solver')
