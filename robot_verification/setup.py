from setuptools import find_packages, setup

package_name = 'robot_verification'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lucas Vieira',
    maintainer_email='lucasfranciscovieira@gmail.com',
    description=(
        'Bench checks: the link check and the pass/fail criteria of the '
        'verifications on ros2 bags.'
    ),
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'check_link = robot_verification.check_link:main',
        ],
    },
)
