#!/bin/bash
#PBS -N JupyterLabConda_Job
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=4gb:scratch_local=10gb:ngpus=1
#PBS -l walltime=4:00:00
#PBS -m ae
# The 4 lines above are options for scheduling system: job will run 4 hours at maximum, 1 machine with 2 processors + 4gb RAM memory + 10gb scratch memory  are requested, email notification will be sent when the job aborts (a) or ends (e) 

# Adapted from /cvmfs/singularity.metacentrum.cz/NGC/JupyterLabPyTorch_Job.sh

echo ${PBS_O_LOGNAME:?This script must be run under PBS scheduling system, execute: qsub $0}

# define variables
MAMBA=/storage/praha1/home/hrabalm/miniforge3/bin/mamba
CONDA_ENV=/storage/praha1/home/hrabalm/envs/unsloth
HOMEDIR=/storage/praha1/home/$USER # substitute username and path to to your real username and path
HOSTNAME=`hostname -f`
PORT="8888"

#find nearest free port to listen
isfree=$(netstat -taln | grep $PORT)
while [[ -n "$isfree" ]]; do
    PORT=$[PORT+1]
    isfree=$(netstat -taln | grep $PORT)
done


# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

#set variables for runtime data
export TMPDIR=$SCRATCHDIR

# move into $HOME directory
cd $HOMEDIR 
if [ ! -f ./.jupyter/jupyter_server_config.json ]; then
   echo "jupyter passwd reset!"
   mkdir -p .jupyter/
   #here you can commem=nt randomly generated password and set your password
   pass=`dd if=/dev/urandom count=1 2> /dev/null | uuencode -m - | sed -ne 2p | cut -c-12` ; echo $pass
   #pass="SecretPassWord" 
   hash=`"$CONDA_ENV"/bin/python -c "from jupyter_server.auth import passwd ; hash = passwd('$pass') ; print(hash)" 2>/dev/null`
   cat > .jupyter/jupyter_server_config.json << EOJson
{
  "ServerApp": {
      "password": "$hash"
    }
}
EOJson
  PASS_MESSAGE="Your password was set to '$pass' (without ticks)."
else
  PASS_MESSAGE="Your password was already set before."
fi

#generate SSL cetificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ./.jupyter/mykey.key -out ./.jupyter/mycert.pem -subj "/CN=$HOSTNAME /O=myOrg /OU=MetaCentrum"

#write SSL config, do not ask for rewrite
cat > .jupyter/jupyter_lab_config.py << EOJsonConfig
c.ServerApp.certfile = u'$HOMEDIR/.jupyter/mycert.pem'
c.ServerApp.keyfile = u'$HOMEDIR/.jupyter/mykey.key'
EOJsonConfig

# MAIL to user HOSTNAME
# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually 
#echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

EXECMAIL=`which mail`
$EXECMAIL -s "JupyterLab with Conda job is running on $HOSTNAME:$PORT" $PBS_O_LOGNAME << EOFmail
Job with JupiterLab with Conda was started.

Use URL  https://$HOSTNAME:$PORT 

$PASS_MESSAGE

You can reset password by deleting file $HOMEDIR/.jupyter/jupyter_lab_config.json and run job again with this script.
EOFmail

export JUPYTER_CONFIG_DIR="$HOMEDIR/.jupyter"
$MAMBA run --prefix="$CONDA_ENV" --no-capture-output jupyter-lab --port $PORT --ip '*' --no-browser
# setting token with parameter  --NotebookApp.token=abcd123456
#singularity exec --nv -H $HOMEDIR $SING_IMAGE jupyter-lab --port $PORT --NotebookApp.token=abcd123456


# clean the SCRATCH directory
clean_scratch
