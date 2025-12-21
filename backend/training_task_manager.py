"""
训练任务管理器。

管理异步训练任务，提供日志捕获和状态查询功能。
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class TaskStatus(Enum):
    """任务状态枚举。"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingTask:
    """训练任务数据类。"""
    
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    logs: List[str] = field(default_factory=list)
    progress_line: str = ""  # 当前进度条（tqdm），单独存储
    error_message: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    
    # 内部使用
    _process: Optional[subprocess.Popen] = field(default=None, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, repr=False)


# 用于过滤的正则表达式
WARNING_PATTERNS = [
    r"FutureWarning:",
    r"UserWarning:",
    r"DeprecationWarning:",
    r"Some weights of the model checkpoint",
    r"This IS expected if you are initializing",
    r"This IS NOT expected if you are initializing",
    r"resume_download.*is deprecated",
    r"127\.0\.0\.1 - -",  # Flask HTTP 请求日志
    r"^\s*warnings\.warn\(",
]

# tqdm 进度条特征
TQDM_PATTERNS = [
    r"\d+%\|",      # 例如 "50%|"
    r"it/s\]?$",     # 例如 "3.37it/s]"
    r"s/it\]?$",     # 例如 "0.30s/it]"
]


def _is_warning_line(line: str) -> bool:
    """检查是否为警告/无用信息行。"""
    for pattern in WARNING_PATTERNS:
        if re.search(pattern, line):
            return True
    return False


def _is_tqdm_line(line: str) -> bool:
    """检查是否为 tqdm 进度条行。"""
    for pattern in TQDM_PATTERNS:
        if re.search(pattern, line):
            return True
    return False


def _clean_line(line: str) -> str:
    """清理行内容（移除回车符等）。"""
    # 移除 ANSI 转义码
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    line = ansi_escape.sub('', line)
    # 移除回车符
    line = line.replace('\r', '')
    return line.strip()


class TrainingTaskManager:
    """训练任务管理器（单例模式）。"""
    
    _instance: Optional[TrainingTaskManager] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> TrainingTaskManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tasks: Dict[str, TrainingTask] = {}
        return cls._instance
    
    def create_task(self) -> TrainingTask:
        """创建新的训练任务。"""
        task_id = str(uuid.uuid4())[:8]
        task = TrainingTask(task_id=task_id)
        self._tasks[task_id] = task
        return task
    
    def get_task(self, task_id: str) -> Optional[TrainingTask]:
        """获取任务。"""
        return self._tasks.get(task_id)
    
    def start_training(
        self,
        task: TrainingTask,
        cmd: List[str],
        cwd: Path,
        on_complete: Optional[Callable[[TrainingTask], None]] = None,
    ) -> None:
        """
        启动训练进程。
        
        Args:
            task: 训练任务
            cmd: 命令行参数
            cwd: 工作目录
            on_complete: 训练完成回调
        """
        def run_training():
            task.status = TaskStatus.RUNNING
            task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 训练开始...")
            task.logs.append(f"命令: {' '.join(cmd)}")
            task.logs.append("-" * 50)
            
            try:
                # 启动子进程
                process = subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,  # 行缓冲
                    encoding='utf-8',
                    errors='replace',
                )
                task._process = process
                
                # 逐行读取输出
                for line in iter(process.stdout.readline, ''):
                    if not line:
                        break
                    
                    cleaned = _clean_line(line)
                    if not cleaned:
                        continue
                    
                    # 过滤警告信息
                    if _is_warning_line(cleaned):
                        continue
                    
                    # 识别 tqdm 进度条
                    if _is_tqdm_line(cleaned):
                        task.progress_line = cleaned
                    else:
                        task.logs.append(cleaned)
                
                process.wait()
                
                if process.returncode == 0:
                    task.status = TaskStatus.COMPLETED
                    task.logs.append("-" * 50)
                    task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 训练完成！")
                else:
                    task.status = TaskStatus.FAILED
                    task.error_message = f"进程退出码: {process.returncode}"
                    task.logs.append(f"❌ 训练失败: {task.error_message}")
                    
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error_message = str(e)
                task.logs.append(f"❌ 错误: {e}")
            finally:
                task.completed_at = datetime.now()
                task.progress_line = ""  # 清空进度条
                if on_complete:
                    on_complete(task)
        
        # 启动线程
        thread = threading.Thread(target=run_training, daemon=True)
        task._thread = thread
        thread.start()
    
    def get_logs_since(self, task_id: str, since_index: int = 0) -> Dict[str, Any]:
        """
        获取指定索引之后的日志。
        
        Args:
            task_id: 任务 ID
            since_index: 起始索引（获取此索引之后的日志）
        
        Returns:
            包含状态、新日志、进度条的字典
        """
        task = self.get_task(task_id)
        if not task:
            return {
                "status": "not_found",
                "logs": [],
                "progress": "",
                "log_index": 0,
            }
        
        new_logs = task.logs[since_index:]
        return {
            "status": task.status.value,
            "logs": new_logs,
            "progress": task.progress_line,
            "log_index": len(task.logs),
            "error": task.error_message if task.status == TaskStatus.FAILED else "",
        }
    
    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """清理超过指定时间的旧任务。"""
        now = datetime.now()
        to_remove = []
        for task_id, task in self._tasks.items():
            if task.completed_at:
                age = (now - task.completed_at).total_seconds() / 3600
                if age > max_age_hours:
                    to_remove.append(task_id)
        
        for task_id in to_remove:
            del self._tasks[task_id]
        
        return len(to_remove)


# 全局实例
task_manager = TrainingTaskManager()
