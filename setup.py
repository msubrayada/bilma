

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(

    name="bilma",  # Required

    version="0.1.8",  # Required

    description="A BERT implementation for twitter in Spanish.",  # Optional

    long_description=long_description,  # Optional
   
    long_description_content_type="text/markdown",  # Optional (see note above)
   
    url="https://github.com/msubrayada/bilma",  # Optional
   
    author="Guillermo Ruiz",  # Optional
    author_email="lgruiz@centrogeo.edu.mx",  # Optional
  
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",

        "Programming Language :: Python :: 3",
       
    ],

    #package_dir={"": "bilma"},  # Optional

    packages=['bilma', 'bilma/test'],
    
    
    #install_requires=["tensorflow >= 2.4"],  # Optional
    
    include_package_data=True,
    zip_safe=False,
    package_data={
        "bilma": ["resources/vocab_file_ALL.txt",                   
                 ],
    },
    
    project_urls={ 
        "Bug Reports": "https://github.com/msubrayada/bilma/issues",
    },
)