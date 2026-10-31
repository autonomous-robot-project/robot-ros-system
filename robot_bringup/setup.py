from setuptools import setup

package_name = 'robot_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/maps', ['maps/m.pgm', 'maps/m.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lucas Vieira',
    maintainer_email='lucasfranciscovieira@gmail.com',
    description=(
        'Launch files and configuration of the robot: the deliberator in the test '
        'configuration of the verifications.'
    ),
    license='Proprietary',
)
