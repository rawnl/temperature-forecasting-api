#from apscheduler.schedulers.background import BackgroundScheduler
from app import create_app, db

app = create_app()

'''
def start_scheduler():
    # define a background schedule
    # Attention: you cannot use a blocking scheduler here as that will block the script from proceeding.
    scheduler = BackgroundScheduler()

    # define your job trigger
    #hourse_keeping_trigger = CronTrigger(hour='12', minute='30')

    # add your job
    #scheduler.add_job(func=run_housekeeping, trigger=hourse_keeping_trigger)

    # start the scheduler
    scheduler.start()
'''

# run server
if __name__ == '__main__':
    #start_scheduler()
    app.run(debug=True)

