all: help

help:
	@echo ""
	@echo "-- Help Menu"
	@echo ""
	@echo "   1. make build              - build the Docker image"
	@echo "   2. make clean              - remove the Docker image"
	@echo ""

# Build the Docker image
build:
	@docker build --tag sam2_realtime -f ./Dockerfile ..


# Remove the Docker image
clean:
	@docker rmi -f sam2_realtime
