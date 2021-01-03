#!/usr/bin/env bash

rsync -av --exclude='*.pt' prince:/path/to/experiment/folder/data/runs/ /local/experiment/folder/data/runs/