from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.ssh.hooks.ssh import SSHHook
from airflow.providers.ssh.operators.ssh import SSHOperator
import pendulum

args = {'owner': 'airflow'}

dag = DAG(dag_id = 'add_new_users', default_args = args, schedule_interval = '50 21 * * *', start_date = pendulum.datetime(2023, 1, 23, tz = "UTC"), tags = ['SSH'])

sshHook = SSHHook(remote_host = '84.201.153.30', username = 'dmitry-ds', key_file = r'/home/dmitryairflow/sshn/id_rsa')

run_add_new_users = SSHOperator(ssh_hook = sshHook, task_id = 'task1', command = 'sudo su - dmitry-ds -c "bash /home/dmitry-ds/rec-sys/Anime-recommender-engine/simulate-new-users/simulate-new-users.sh" ', dag=dag)

run_add_new_users