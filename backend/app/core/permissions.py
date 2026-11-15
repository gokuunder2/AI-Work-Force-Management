# enum: Python's tool for a fixed list of allowed values
import enum

from app.models.enums import Role


# Every action the system can allow or deny.
# Routes ask for one of these, never for a role name.
class Permission(str, enum.Enum):
    # Read your own employee record
    EMPLOYEE_READ_OWN = "employee:read_own"
    # Read the records of your team (managers)
    EMPLOYEE_READ_TEAM = "employee:read_team"
    # Read every employee record (HR, Admin)
    EMPLOYEE_READ_ALL = "employee:read_all"
    # Create or edit employee records
    EMPLOYEE_WRITE = "employee:write"
    # View salary data (a separate door from employee data)
    PAYROLL_READ = "payroll:read"
    # Change salary data
    PAYROLL_WRITE = "payroll:write"
    # Create, disable and edit user accounts
    USER_MANAGE = "user:manage"
    # Change someone's role
    ROLE_CHANGE = "role:change"
    # Export employee data (always audited)
    EXPORT_EMPLOYEES = "export:employees"
    # Run the task allocation engine
    ALLOCATION_RUN = "allocation:run"
    # Use the CRM / leads module (the only thing Sales gets)
    CRM_ACCESS = "crm:access"


# A short alias so the map below stays readable
P = Permission

# The ONE place that says what each role may do. Anything not listed is denied.
ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    # Admin gets every permission in the enum
    Role.ADMIN: frozenset(Permission),
    Role.HR: frozenset({
        P.EMPLOYEE_READ_ALL, P.EMPLOYEE_WRITE, P.PAYROLL_READ, P.PAYROLL_WRITE,
        P.USER_MANAGE, P.EXPORT_EMPLOYEES, P.ALLOCATION_RUN,
    }),
    Role.MANAGER: frozenset({
        P.EMPLOYEE_READ_OWN, P.EMPLOYEE_READ_TEAM, P.ALLOCATION_RUN,
    }),
    # Employees can only read their own record
    Role.EMPLOYEE: frozenset({P.EMPLOYEE_READ_OWN}),
    # Sales gets CRM only: no employee or payroll permission at all
    Role.SALES: frozenset({P.CRM_ACCESS}),
}