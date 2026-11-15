# enum: Python's built-in tool for a fixed list of allowed values
import enum


# The only roles that can exist. The database rejects anything else.
# "str" is included so a Role can be compared with and stored as plain text.
class Role(str, enum.Enum):
    # Left side = name used in code, right side = text stored in the database
    ADMIN = "admin"
    HR = "hr"
    MANAGER = "manager"
    EMPLOYEE = "employee"
    SALES = "sales"  # CRM only: denied all employee and payroll data


# The only employment statuses an employee row can have
class EmployeeStatus(str, enum.Enum):
    ACTIVE = "active"
    ON_LEAVE = "on_leave"
    TERMINATED = "terminated"