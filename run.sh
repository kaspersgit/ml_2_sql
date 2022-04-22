#!/bin/bash
# Let's call it ML 2 SQL (made at http://www.kammerl.de/ascii/AsciiSignature.php)
cat << 'EOF'



`7MMM.     ,MMF'`7MMF'                        .M"""bgd   .g8""8q. `7MMF'
  MMMb    dPMM    MM                         ,MI    "Y .dP'    `YM. MM
  M YM   ,M MM    MM             pd*"*b.     `MMb.     dM'      `MM MM
  M  Mb  M' MM    MM            (O)   j8       `YMMNq. MM        MM MM
  M  YM.P'  MM    MM      ,         ,;j9     .     `MM MM.      ,MP MM      ,
  M  `YM'   MM    MM     ,M      ,-='        Mb     dM `Mb.    ,dP' MM     ,M
.JML. `'  .JMML..JMMmmmmMMM     Ammmmmmm     P"Ybmmd"    `"bmmd"' .JMMmmmmMMM
                                                             MMb
                                                              `bood'

EOF

printf "\n\n"

# Check if AWS cli and aws login tool are installed
printf "Package versions:\n"
# pip list | grep -F sklearn || { printf "sklearn not installed, please install by running: pip install sklearn"; exit 1; }
# pip list | grep -F matplotlib || { printf "matplotlib not installed, please install by running: pip install matplotlib"; exit 1; }
# pip list | grep -F imbalanced-learn || { printf "imblearn not installed, please install by running: pip install imblearn"; exit 1; }
# pip list | grep -F numpy || { printf "numpy not installed, please install by running: pip install numpy"; exit 1; }
# pip list | grep -F pandas || { printf "pandas not installed, please install by running: pip install pandas"; exit 1; }


# select data
# Bash Menu Script
printf "Select path to csv file for training the model\n"
source bash_utils/bash_menu.sh input/data .csv
printf "\nCSV file $FILEPATH will be used for modelling\n"
CSVPATH=$FILEPATH

# check if file path is s3 path and download file if that is the case
if [[ "${CSVPATH}" == "s3://"*".csv" ]]; then
  printf "${CSVPATH} is a CSV on an S3 location"
  aws s3 cp ${CSVPATH} input/data/. --profile iam-sync/lakehouse-redshift/lakehouse-redshift.IdP_risk_general_eu@klarna-data-production || printf "Install AWS CLI by following: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html" && exit
elif [[ "${CSVPATH}" == "s3://"* ]]; then
  printf "${CSVPATH} is an S3 location but not a CSV file"
  aws s3 cp ${CSVPATH} input/data/. --profile iam-sync/lakehouse-redshift/lakehouse-redshift.IdP_risk_general_eu@klarna-data-production || printf "Install AWS CLI by following: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html" && exit
else
  printf "${CSVPATH} is not an S3 location"
fi
# aws s3 cp s3://bucket/folder/file.txt .

# select target and feature columns
# Bash Menu Script
printf "\n\nSelect path to csv file for training the model\n"
source bash_utils/bash_menu.sh input/configuration .json
printf "\nJson file $FILEPATH will be used for modelling\n"
JSONPATH=$FILEPATH

# Selecting what type of output we are looking for
# https://askubuntu.com/questions/1705/how-can-i-create-a-select-menu-in-a-shell-script
printf "\nWhat type of model do you want? \n"
PS3="Choose a number: "
types=("Decision tree" "Explainable boosting machine")
select type in "${types[@]}"
do
    case $type in
        "Decision tree")
            printf "You chose $type \n\n"
            MODEL_TYPE='decision_tree'
            break
            ;;
        "Explainable boosting machine")
            printf "You chose $type \n\n"
            MODEL_TYPE='ebm'
            break
            ;;
        *) printf "invalid option $REPLY \n";;
    esac
done

# Set name of model
printf "\nGive it a name:"
read MODELNAME

# Set current data
CURRENT_DATE=$(date +%Y%m%d)
FULL_MODEL_NAME=${CURRENT_DATE}_${MODELNAME}

# Make directory with current data and model name
mkdir trained_models/${FULL_MODEL_NAME} || exit
mkdir -p trained_models/${FULL_MODEL_NAME}/feature_importance
mkdir -p trained_models/${FULL_MODEL_NAME}/performance
mkdir -p trained_models/${FULL_MODEL_NAME}/model

printf "Starting script to create model"

# Run python script with given input
printf "\npython3 main.py --name trained_models/${FULL_MODEL_NAME} --data_path $CSVPATH --configuration $JSONPATH --output_type $OUTPUT_TYPE --model $MODEL_TYPE"
python3 main.py --name trained_models/${FULL_MODEL_NAME} --data_path $CSVPATH --configuration $JSONPATH --model $MODEL_TYPE

printf "\n\nModel outputs can be found in folder: \n${PWD}/trained_models/${FULL_MODEL_NAME}"
