## Metrics
* Time to AUC or BSS threshold on validation data

## Summer School Checklist
* Jupyter notebook describing GOES problem and data
* Notebook will go in AI4ESS hackathon repository: https://github.com/NCAR/ai4ess-hackathon-2020
* Key components:
    * Data downloading function from AWS (Not Done)
    * Data loading function (Done)
    * Split data into training, validation and testing sets (in script not function)
      * Arguments: start and end dates for training period, validation period, and testing period, input data, input labels
      * Example of data loading script with date evaluation (https://github.com/djgagne/HWT_mode/blob/master/hwtmode/data.py#L9)
      * Returns separate xarray DataArrays for training, validation, and testing for inputs and outputs.
      * Use different days within each month for training validation and testing 
    * Data rescaling function (Mostly Done)
    * Re-run run_extract_goes16_hpss.sh (Not Done)
        * Extract data from March 1, 2019 through September 30, 2019
    * Re-run process_goes16.py with lead_time: "60Min" (Not done)
    * Re-run processing for 32 and 64 but not 128
    