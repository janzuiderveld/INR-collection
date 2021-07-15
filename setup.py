from distutils.core import setup
setup(
  name = 'INR_collection',
  packages = ['INR_collection'],
  version = '0.13',
  license='MIT',
  description = 'A collection of conditional \ modulatable implicit neural representation implementations and basic building blocks in PyTorch.',
  author = 'Jan Zuiderveld',
  author_email = 'janzuiderveld@gmail.com',
  url = 'https://github.com/janzuiderveld/INR-collection',
  download_url = 'https://github.com/janzuiderveld/INR-collection/archive/v_01.tar.gz',
  keywords = ['Implicit', 'neural', 'representation', 'SIREN', 'functional', 'IM-NET'],
  install_requires=[
          'numpy',
          'torch'
      ],
  classifiers=[

    'Development Status :: 4 - Beta',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',

    # Pick your license as you wish
    'License :: OSI Approved :: MIT License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
