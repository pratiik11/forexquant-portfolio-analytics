# Setting Up Automated Daily Updates

## Windows Instructions

1. Open Windows Task Scheduler:
   - Press `Win+R`, type `taskschd.msc`, and press Enter

2. Create a new task:
   - In the right panel, click on "Create Basic Task"
   - Name: "ForexQuant Daily Update"
   - Description: "Updates currency pair data daily"
   - Click "Next"

3. Set the trigger:
   - Select "Daily"
   - Click "Next"
   - Set the start time to a time when your computer will be on (e.g., 9:00 AM)
   - Click "Next"

4. Select the action:
   - Choose "Start a program"
   - Click "Next"

5. Set the program details:
   - Program/script: Browse and select the `update_forex_data.bat` file
   - Start in: Your project directory
   - Click "Next"

6. Complete:
   - Check "Open the Properties dialog for this task when I click Finish"
   - Click "Finish"

7. In the Properties dialog:
   - Go to the "Conditions" tab
   - Check "Wake the computer to run this task" if desired
   - Go to the "Settings" tab
   - Check "Run task as soon as possible after a scheduled start is missed"
   - Click "OK"

Now the task will run automatically every day at the specified time. The update results will be logged in `update_log.txt`. 