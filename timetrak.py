from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import argparse
import sys
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class TaskStatus(Enum):
    """Enumeration of possible task statuses."""

    ACTIVE = "active"
    COMPLETED = "completed"


@dataclass
class Task:
    """Data class representing a time tracking task."""

    start_time: datetime
    stop_time: Optional[datetime] = None
    project: Optional[str] = None

    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate task duration if the task is completed."""
        if self.stop_time:
            return self.stop_time - self.start_time
        return None

    @property
    def status(self) -> TaskStatus:
        """Determine if the task is active or completed."""
        return TaskStatus.COMPLETED if self.stop_time else TaskStatus.ACTIVE

    def to_log_entry(self) -> str:
        """Convert task to log file entry format."""
        entry = f"Start: {self.start_time.strftime('%Y-%m-%d %H:%M:%S.%f')}"
        if self.stop_time:
            entry += f", Stop: {self.stop_time.strftime('%Y-%m-%d %H:%M:%S.%f')}"
        if self.project:
            entry += f", Project: {self.project}"
        return entry

    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate task duration if the task is completed."""
        if self.stop_time:
            return self.stop_time - self.start_time
        return None


class TimeKeeper:
    """Main class for handling time tracking operations."""

    def __init__(self, log_file: Path | None = None):
        """
        Initialize TimeKeeper with log file path.

        Args:
            log_file: Path to the log file
        """
        self.log_file = Path(log_file)
        self._ensure_log_file_exists()

    def _ensure_log_file_exists(self) -> None:
        """Create log file if it doesn't exist."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            self.log_file.touch()

    def _parse_log_entry(self, line: str) -> Task:
        """
        Parse a log file entry into a Task object.

        Args:
            line: Raw log entry string

        Returns:
            Task object representing the entry

        Raises:
            ValueError: If log entry format is invalid
        """
        parts = [p.strip() for p in line.split(",")]
        task_dict = {}

        for part in parts:
            if ":" not in part:
                raise ValueError(f"Invalid log entry format: {line}")
            key, value = [p.strip() for p in part.split(":", 1)]
            task_dict[key.lower()] = value

        try:
            task = Task(
                start_time=datetime.strptime(
                    task_dict["start"], "%Y-%m-%d %H:%M:%S.%f"
                ),
                stop_time=(
                    datetime.strptime(task_dict["stop"], "%Y-%m-%d %H:%M:%S.%f")
                    if "stop" in task_dict
                    else None
                ),
                project=task_dict.get("project"),
            )
            return task
        except (KeyError, ValueError) as e:
            raise ValueError(f"Error parsing log entry: {e}")

    def find_active_task(
        self, project: Optional[str] = None
    ) -> Optional[Tuple[int, Task]]:
        """
        Find the first active task matching the project filter.

        Args:
            project: Optional project name filter

        Returns:
            Tuple of (line number, Task) if found, None otherwise
        """
        with self.log_file.open("r") as f:
            for idx, line in enumerate(f):
                try:
                    task = self._parse_log_entry(line.strip())
                    if task.status == TaskStatus.ACTIVE:
                        if not project or task.project == project:
                            return idx, task
                except ValueError:
                    logger.warning(f"Skipping invalid log entry at line {idx + 1}")
        return None

    def find_all_tasks(self, project: Optional[str] = None) -> List[Task]:
        """
        Find all tasks matching the project filter.

        Args:
            project: Optional project name filter

        Returns:
            List of Task objects
        """
        tasks = []
        with self.log_file.open("r") as f:
            for idx, line in enumerate(f):
                try:
                    task = self._parse_log_entry(line.strip())
                    if project and task.project != project:
                        continue
                    if task.status == TaskStatus.ACTIVE:
                        tasks.append(task)
                except ValueError:
                    logger.warning(f"Skipping invalid log entry at line {idx + 1}")
        return tasks

    def start_task(self, project: Optional[str] = None) -> None:
        """
        Start a new task.

        Args:
            project: Optional project name

        Raises:
            RuntimeError: If there's already an active task
        """
        if self.find_active_task(project):
            raise RuntimeError("Task already in progress")

        task = Task(start_time=datetime.now(), project=project)
        with self.log_file.open("a") as f:
            f.write(f"{task.to_log_entry()}\n")
        logger.info(
            "Started new task" + (f" for project '{project}'" if project else "")
        )

    def stop_task(self, project: Optional[str] = None) -> None:
        """
        Stop the active task.

        Args:
            project: Optional project name to match

        Raises:
            RuntimeError: If no active task is found
        """
        result = self.find_active_task(project)
        if not result:
            raise RuntimeError("No task in progress")

        idx, task = result
        task.stop_time = datetime.now()

        with self.log_file.open("r") as f:
            lines = f.readlines()

        lines[idx] = task.to_log_entry() + "\n"

        with self.log_file.open("w") as f:
            f.writelines(lines)

        logger.info(f"Stopped task. Duration: {delta_to_str(task.duration)}")

    def get_tasks(self, project: Optional[str] = None) -> List[Task]:
        """
        Get all tasks, optionally filtered by project.

        Args:
            project: Optional project name filter

        Returns:
            List of Task objects
        """
        tasks = []
        with self.log_file.open("r") as f:
            for idx, line in enumerate(f):
                try:
                    task = self._parse_log_entry(line.strip())
                    if not project or task.project == project:
                        tasks.append(task)
                except ValueError:
                    logger.warning(f"Skipping invalid log entry at line {idx + 1}")
        return tasks

    def clear_tasks(self, project: Optional[str] = None) -> None:
        lines = self.log_file.read_text().splitlines()
        kept = []
        for line in lines:
            if project and f"Project: {project}" in line:
                continue
            if not project:
                continue
            kept.append(line)
        self.log_file.write_text("\n".join(kept) + ("\n" if kept else ""))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Time tracking CLI tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "action",
        choices=["start", "stop", "status", "info", "clear"],
        help="Action to perform",
    )
    parser.add_argument("--project", "-p", help="Project name for filtering tasks")
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        nargs="?",
        default=None,
        help="Path to log file",
    )
    return parser.parse_args()


def delta_to_str(t: timedelta) -> str:
    """Convert a timedelta to a human-readable string."""
    days, hours, minutes, seconds = (
        t.days,
        t.seconds // 3600,
        (t.seconds // 60) % 60,
        t.seconds % 60,
    )
    parts = []
    if days:
        parts.append(f"{days} days")
    if hours:
        parts.append(f"{hours} hours")
    if minutes:
        parts.append(f"{minutes} minutes")
    if seconds:
        parts.append(f"{seconds} seconds")
    return ", ".join(parts) if parts else "0 seconds"


def temp_path() -> Path:
    return Path(tempfile.gettempdir()) / "_timetrak.log"


def main() -> None:
    """Main entry point for the CLI application."""
    try:
        args = parse_args()
        keeper = TimeKeeper(args.file or temp_path())

        if args.action == "start":
            keeper.start_task(args.project)

        elif args.action == "stop":
            keeper.stop_task(args.project)

        elif args.action == "status":
            results = keeper.find_all_tasks(args.project)
            # if result:
            #     _, task = result
            #     print(f"Task in progress: {task.to_log_entry()}")
            # else:
            #     print("No task in progress")
            if results:
                if args.project:
                    print(f"\nAll tasks for project '{args.project}':")
                else:
                    print("\nAll tasks:")
                for task in results:
                    print(task.to_log_entry())
            else:
                print("No tasks found")

        elif args.action == "info":
            tasks = keeper.get_tasks(args.project)
            active_tasks = [t for t in tasks if t.status == TaskStatus.ACTIVE]
            completed_tasks = [t for t in tasks if t.status == TaskStatus.COMPLETED]

            total_time: timedelta = sum(
                (t.duration for t in completed_tasks), start=timedelta()
            )

            print("\nCompleted tasks:")
            for task in completed_tasks:
                print(task.to_log_entry(), end="")
                dur = task.duration
                if dur:
                    print(f" ({delta_to_str(dur)})")
                else:
                    print("")

            print(f"\nTotal time: {delta_to_str(total_time)}")

            if active_tasks:
                print("\nActive tasks:")
                for task in active_tasks:
                    print(task.to_log_entry())
        elif args.action == "clear":
            keeper.clear_tasks(args.project)
            print("Cleared tasks" + (f" for '{args.project}'" if args.project else ""))

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
