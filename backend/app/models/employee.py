# uuid: used for the type of the random IDs
import uuid
# date: the Python type for a calendar date (hire date)
from datetime import date
# Decimal: exact numbers for money (floats round badly)
from decimal import Decimal

# Column types and rules we use when defining the tables
from sqlalchemy import CheckConstraint, Date, Enum, ForeignKey, Numeric, String
# PostgreSQL's native UUID column type
from sqlalchemy.dialects.postgresql import UUID
# Mapped/mapped_column: define typed columns; relationship: link tables together
from sqlalchemy.orm import Mapped, mapped_column, relationship

# Base: parent of all tables; the two mixins add the id and created/updated columns
from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
# The allowed employee statuses (active, on_leave, terminated)
from app.models.enums import EmployeeStatus


# DEPARTMENTS TABLE: gets a random UUID id and created_at/updated_at automatically
class Department(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    # The real table name in PostgreSQL
    __tablename__ = "departments"

    # Department name, max 100 characters. unique=True: no duplicate names.
    # nullable=False: a name is required.
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)


# EMPLOYEES TABLE: HR details only. Salary is deliberately NOT here.
class Employee(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    # The real table name in PostgreSQL
    __tablename__ = "employees"

    # First name, required, up to 100 characters
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    # Last name, required, up to 100 characters
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    # Work email, required, and no two employees can share one
    work_email: Mapped[str] = mapped_column(String(254), unique=True, nullable=False)
    # Job title is optional ("str | None" means it may be empty)
    job_title: Mapped[str | None] = mapped_column(String(150))
    # Hire date is optional
    hire_date: Mapped[date | None] = mapped_column(Date)
    # Employment status, stored as text. Only the values in EmployeeStatus are accepted.
    status: Mapped[EmployeeStatus] = mapped_column(
        # native_enum=False stores plain text (easier to change later);
        # values_callable stores "active" rather than the Python name "ACTIVE"
        Enum(EmployeeStatus, native_enum=False, length=20,
             values_callable=lambda e: [m.value for m in e]),
        # New employees start as active unless told otherwise
        default=EmployeeStatus.ACTIVE,
        nullable=False,
    )

    # Which department this employee belongs to (optional)
    department_id: Mapped[uuid.UUID | None] = mapped_column(
        # ForeignKey: must point at a real department.
        # ondelete="SET NULL": if the department is deleted, keep the employee, clear the link.
        # index=True: makes "all employees in a department" lookups fast.
        UUID(as_uuid=True), ForeignKey("departments.id", ondelete="SET NULL"), index=True
    )
    # Direct reporting line, used for manager scoping. Points to another employee row.
    manager_id: Mapped[uuid.UUID | None] = mapped_column(
        # If the manager row is deleted, the link is cleared instead of deleting the report
        UUID(as_uuid=True), ForeignKey("employees.id", ondelete="SET NULL"), index=True
    )

    # Lets code write employee.department to get the Department object
    department: Mapped[Department | None] = relationship()
    # Lets code write employee.compensation to get the salary row (loaded only when asked)
    compensation: Mapped["EmployeeCompensation | None"] = relationship(
        # back_populates: the other side of the link is EmployeeCompensation.employee.
        # uselist=False: one salary row per employee, not a list.
        # cascade="all, delete-orphan": deleting an employee deletes their salary row too.
        back_populates="employee", uselist=False, cascade="all, delete-orphan"
    )


# EMPLOYEE_COMPENSATION TABLE: payroll in its own table, so a normal employee
# query can never return salary by accident, and it can have its own permission
class EmployeeCompensation(TimestampMixin, Base):
    """Payroll data lives apart from Employee so it needs its own permission
    and can never leak through a generic employee query."""

    # The real table name in PostgreSQL
    __tablename__ = "employee_compensation"
    # A database-level rule: salary can never be negative, even if the app has a bug
    __table_args__ = (CheckConstraint("base_salary >= 0", name="salary_non_negative"),)

    # Primary key AND foreign key: exactly one compensation row per employee
    employee_id: Mapped[uuid.UUID] = mapped_column(
        # ondelete="CASCADE": if the employee is deleted, their salary row goes too
        UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"), primary_key=True
    )
    # Salary amount: up to 12 digits total, 2 after the decimal point. Required.
    base_salary: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    # 3-letter currency code, defaults to INR
    currency: Mapped[str] = mapped_column(String(3), default="INR", nullable=False)
    # How often the employee is paid, defaults to monthly
    pay_frequency: Mapped[str] = mapped_column(String(20), default="monthly", nullable=False)

    # Lets code write compensation.employee to get the Employee object
    employee: Mapped[Employee] = relationship(back_populates="compensation")