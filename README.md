# Near-Far Pile Algorithm for GPU

This is a naive implementation of the Near-Far Pile Single Source Shortest Path Algorithm
by [Davidson et al.](http://escholarship.org/uc/item/8qr166v2)

## Requirements

- CUDA 7.5
- Thrust 1.8.1
- [CUSP 0.5.1](https://cusplibrary.github.io/)

## Build

### Using Docker

1. Install [Docker and NVIDIA for Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
2. Build image `docker build . --pull --rm f "Dockerfile" -t naive-near-far-gpu:latest`
3. Image will automatically download libraries and compile the software

### Locally

1. Make sure CUDA 7.5 is installed
2. Run the build script `sh build.sh`
3. The build script downloads CUSP

## Running

- Command: `` ./main `[m|d]` <file> ``
- Example `./main m graph.mtx`
- First argument: file type. `m` for Matrix Market \*.mtx files, `d` for Dimacs
- Second argument: path to file

### Docker

1. Create a local folder to put your graphs
2. Set this local folder as a volume in Docker
3. Run command

```
docker run -v <local-path-to-graphs>:/<docker-path> --name near-far --gpus all near-far-pile:latest ./main `[m|d]` <docker-path>/<file>
```

4. Copy "distance.mtx" file from Docker to local folder: `docker cp near-far:distance.mtx .`

## License

This project is licensed under GPLv3
