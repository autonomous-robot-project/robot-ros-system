from setuptools import find_packages, setup

package_name = 'robot_sequencer'

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
        'Sequencer: runs missions of primitive behaviors on the controller and reacts to its '
        'conditions.'
    ),
    license='Proprietary',
    tests_require=['pytest'],
)
