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
        'Bench checks and measurements: the link check, the link load and the pass/fail '
        'criteria of the verifications on ros2 bags.'
    ),
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'check_link = robot_verification.check_link:main',
            'measure_link_load = robot_verification.measure_link_load:main',
            'measure_clock_drift = robot_verification.measure_clock_drift:main',
            'check_v06 = robot_verification.check_v06:main',
            'check_v02 = robot_verification.check_v02:main',
            'check_v03 = robot_verification.check_v03:main',
            'check_v04 = robot_verification.check_v04:main',
            'check_v05 = robot_verification.check_v05:main',
            'check_v01 = robot_verification.check_v01:main',
            'check_v07 = robot_verification.check_v07:main',
            'run_v02 = robot_verification.run_v02:main',
            'run_v03 = robot_verification.run_v03:main',
            'run_v01 = robot_verification.run_v01:main',
            'run_v04 = robot_verification.run_v04:main',
            'run_v07 = robot_verification.run_v07:main',
            'run_closed_loop = robot_verification.run_closed_loop:main',
            'measure_closed_loop = robot_verification.measure_closed_loop:main',
            'plot_closed_loop = robot_verification.plot_closed_loop:main',
            'plot_clock_drift = robot_verification.plot_clock_drift:main',
            'plot_recording = robot_verification.plot_recording:main',
            'plot_v01 = robot_verification.plot_v01:main',
        ],
    },
)
