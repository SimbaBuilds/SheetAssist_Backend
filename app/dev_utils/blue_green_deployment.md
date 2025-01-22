Use a Blue/Green Deployment (Safest Method)
	1.	Clone the Environment:
eb clone fastapienv04
This creates a new environment with the same configuration but isolated from the live application.
	2. Set up SSH in the cloned environment
    3.	Deploy Code to the Cloned Environment:
eb deploy fastapienv04c3
Test the application thoroughly in the new environment.
	4.	Swap Environments:
	•	Once the new environment is verified, swap it with the old environment:
eb swap fastapienv04 --destination-name fastapienv04c3
