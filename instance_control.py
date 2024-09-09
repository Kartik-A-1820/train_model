import boto3

def stop_instance(instance_id, region_name='ap-south-1'):
    try:
        # Create an EC2 client with the specified region
        ec2 = boto3.client('ec2', region_name=region_name)

        print(f"Attempting to stop instance {instance_id}...")
        
        # Stop the instance
        ec2.stop_instances(InstanceIds=[instance_id])
        
        print(f"Instance {instance_id} has been stopped successfully.")
    except Exception as e:
        print(f"Failed to stop the instance {instance_id}. Error: {e}")
