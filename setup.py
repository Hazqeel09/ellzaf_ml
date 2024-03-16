from setuptools import setup, find_packages

setup(
  name = 'ellzaf_ml',
  packages = find_packages(),
  version = '1.4.13',
  license='MIT',
  description = 'Ellzaf ML',
  long_description_content_type = 'text/markdown',
  author = 'Hazqeel Afyq',
  author_email = 'hazqeel9@gmail.com',
  url = 'https://github.com/Hazqeel09/ellzaf_ml',
  keywords = [
    'artificial intelligence',
    'machine learning',
  ],
  install_requires=[
    'einops>=0.7.0',
    'torch>=1.10',
    'torchvision',
    'mediapipe',
    'opencv-python',
    'scipy',
    'timm>=0.9.0',
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest',
    'torch==1.12.1',
    'torchvision==0.13.1'
  ],
  python_requires=">=3.9",
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8',
  ],
)