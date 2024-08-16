# Coffee Chat Algorithm
This project contains a matching algorithm for pairing employees for professional development initiatives. 

## Getting Started

To run the matching algorithm:

1. Replace the input_file in 'brinda-match.py' main function with the newest CSV file of the month (should be added to 'input' directory)
2. Ensure the outputs of the previous month are in the 'output' folder, named as "deduplicated-YEAR-MONTH-01-profiles.xlsx.csv" 
3. Update the name of the output filr in the 'results.to_csv()' function at the bottom of the main function in 'brinda-match.py'
4. Run 'brinda-match.py'

## Important Notes

- **Manager Matching** The algorithm currently does not filter out managers from potential matches. Potential solution exists in 'brinda-norep_match.py'
- **Vectotization** For improved performance, consider some vectorizing solutions in 'brinda-vectorupdate.ipynb'

## Files

- 'brinda-match.py' : Main script for running the matching algorithm
- 'brinda-norep_matching.py" : Was made to match employees randomly removing previous matches and managers in case 'brinda-match.py' doesn't work.
    - Contains potential code to filter out managers in main function
- 'brinda-vectorupdate.ipynb' : Contains vectorized version of calculate_similarity function for improved efficiency 

