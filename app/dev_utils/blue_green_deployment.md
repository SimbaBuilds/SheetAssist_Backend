Use a Blue/Green Deployment (Safest Method)
	1.	Clone the Environment:
eb clone fastapienv04c3 -n fastapienv04c5
This creates a new environment with the same configuration but isolated from the live application.
	2. Set up SSH in the cloned environment
    3.	Deploy Code to the Cloned Environment:
eb deploy fastapienv04c5

