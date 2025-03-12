# Instructions

1. Generated the Docker image skipping the "pip install -e ." and models checkpoints download;
2. Did this command to "xhost +local:docker" to allow Docker Container to Access X11. Please do "xhost -local:docker" to revoke access;
3. Applied some changes to the docker container and save it into a new image with the command: "
docker commit 64e13905c460  sam2_realtime_saved:v1"
4. Changed the file "sam2_realtime.sh" to run right docker image


# Docker
Attach to the running container
docker exec -it <container_name_or_id> bash

Save container into new image
docker commit <CONTAINER_ID> <new_image_name>:<tag>
docker commit 4014408ef9cb  sam2_realtime_saved:v2

# Github
List remote connections to Github
git remote -v

Pull from the original repo
git pull origin main

Push
git push origin main  # Push to the original repository
git push mysam2 main  # Push to your new repository


