# Instructions

1. Generated the Docker image skipping the "pip install -e ." and models checkpoints download;
2. Did this command to "xhost +local:docker" to allow Docker Container to Access X11. Please do "xhost -local:docker" to revoke access;
3. Applied some changes to the docker container and save it into a new image with the command: "
docker commit 64e13905c460  sam2_realtime_saved:v1"
4. Changed the file "sam2_realtime.sh" to run right docker image


Attach to the running container
docker exec -it <container_name_or_id> bash
