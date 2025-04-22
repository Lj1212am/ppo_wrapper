from setuptools import find_packages, setup

package_name = 'ppo_wrapper'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lee',
    maintainer_email='kevinpaulose05',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ppo_wrapper = ppo_wrapper.ppo_wrapper:main',
            'waypoint_pub = ppo_wrapper.waypoint_pub:main'
        ],
    },
)
