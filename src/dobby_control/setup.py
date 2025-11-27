from setuptools import find_packages, setup

package_name = 'dobby_control'

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
    maintainer='krh',
    maintainer_email='krh@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'move_rover = dobby_control.move_rover:main',
            'yolo_detect = dobby_control.yolo_detect:main',
            'dobby_gesture = dobby_control.dobby_gesture:main',
            'dobby_vision = dobby_control.dobby_vision:main',
            'dobby_control = dobby_control.dobby_control:main',
        ],
    },
)
