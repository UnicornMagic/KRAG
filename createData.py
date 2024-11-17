import os

# Create a directory for simulated data
os.makedirs("data", exist_ok=True)

# Sample content for documents
documents = {
    "leave_policies.txt": """\
Company Leave Policies:
- Employees are entitled to 20 days of annual leave.
- Sick leave requires a doctor's note if more than 2 days.
- Parental leave is granted for up to 12 weeks per year.""",
    
    "agile_practices.txt": """\
Agile Way of Working:
- Daily stand-ups should last no more than 15 minutes.
- Sprint duration is 2 weeks with a focus on deliverables.
- Retrospectives are mandatory to reflect on team processes.""",
    
    "tool_setup.md": """\
Tool Setup Guide:
1. Install the engineering tool suite from the company portal.
2. Configure your environment variables as described in the wiki.
3. Reach out to IT support for VPN configuration.""",
    
    "api_doc.txt": """\
API Documentation:
- Endpoint: POST /api/v1/create-user
  Description: Creates a new user in the system.
  Required Parameters: name, email, password.

- Endpoint: GET /api/v1/user-profile
  Description: Retrieves user profile information.
  Required Headers: Authorization token.""",
}

# Write content to files
for filename, content in documents.items():
    with open(os.path.join("data", filename), "w") as f:
        f.write(content)

print("Sample data created in the 'data' folder!")