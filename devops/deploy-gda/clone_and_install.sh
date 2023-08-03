#!/bin/bash
INSTALL_DIR="/databricks/data"
DATALAKE_BASE_DIR_STAGE="/dbfs/mnt/refined/kedro_h2o_det_dev"
DATALAKE_BASE_DIR_TEST="/dbfs/mnt/refined/kedro_h2o_det_dev"
DATALAKE_BASE_DIR_DEV="/dbfs/mnt/refined/kedro_h2o_det_dev"
DATALAKE_BASE_DIR_DATAENG="/dbfs/mnt/refined/kedro_h2o_det_dev"

ENV=$1

STAGE="stage"
TEST="test"
DEV="dev"
DATAENG='dataeng'

if [[ "$ENV" == "$STAGE" ]]; then
    BASE_DIR=$DATALAKE_BASE_DIR_STAGE
elif [[ "$ENV" == "$TEST" ]]; then
    BASE_DIR=$DATALAKE_BASE_DIR_TEST
elif [[ "$ENV" == "$DATAENG" ]]; then
    BASE_DIR=$DATALAKE_BASE_DIR_DATAENG
else
    BASE_DIR=$DATALAKE_BASE_DIR_DEV
fi

cd $INSTALL_DIR/repo
pip install -r src/requirements.txt
cp data/09_data_dictionaries/* $BASE_DIR/data/09_data_dictionaries/
rm -rf data
ln -s $BASE_DIR/data .

