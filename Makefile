mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

.PHONY: init

init:
	ln -s /content/drive/MyDrive/IoT/data ${mkfile_dir}data