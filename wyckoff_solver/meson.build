dioph_modulename = 'dioph'
dioph_src = ['kinds.f90', 'dioph.f90']
dioph_source = custom_target(dioph_modulename + 'module.c',
  input : dioph_src,  # .f so no F90 wrappers
  output : [
    dioph_modulename + 'module.c',
    dioph_modulename + '-f2pywrappers2.f90'
  ],
  command : [py3, '-m', 'numpy.f2py', '@INPUT@',
    '-m', dioph_modulename,
    '--lower',
    '--build-dir', 'wyckoff_solver',
  ],
)
py3.extension_module(dioph_modulename,
  dioph_src + [dioph_source],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : py_dep,
  install : true,
  subdir: 'wyckoff_solver',
)

py3.install_sources(
  [
    'cache.py',
    'linprog.py',
    'cli.py',
    'spacegroup.py',
    'wyckcomb.py',
  ],
  subdir: 'wyckoff_solver',
)

subdir('utils')
subdir('resources')