#!/usr/bin/env python3
"""
DuelStore - Manages the modular directory structure for prompts, metrics, and experiments
"""

import yaml
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid


@dataclass
class PromptVersion:
    prompt: str
    version: str
    created: str
    parent: Optional[str] = None
    changes: Optional[str] = None
    tags: Optional[List[str]] = None


@dataclass
class PromptMetadata:
    name: str
    description: str
    category: str
    author: str
    created: str
    latest_version: str


@dataclass
class Metric:
    name: str
    type: str
    description: str
    implementation: str
    parameters: Dict[str, Any]
    prompts: Optional[Dict[str, str]] = None


@dataclass
class Experiment:
    experiment_id: str
    name: str
    created: str
    prompts: Dict[str, str]
    metrics: List[str]
    test_cases: List[Dict[str, Any]]
    model: str
    system_prompt: Optional[str] = None


class DuelStore:
    def __init__(self, base_dir: str = ".duel"):
        self.base_dir = Path(base_dir)
        self.prompts_dir = self.base_dir / "prompts"
        self.metrics_dir = self.base_dir / "metrics"
        self.experiments_dir = self.base_dir / "experiments"
        self.history_dir = self.base_dir / "history"
        self.prompt_history_file = self.history_dir / "prompt_evolution.json"
        self.experiment_history_file = self.history_dir / "experiment_history.json"
        
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create all necessary directories if they don't exist."""
        for directory in [self.base_dir, self.prompts_dir, self.metrics_dir, 
                         self.experiments_dir, self.history_dir]:
            directory.mkdir(exist_ok=True)
    
    def _log_prompt_history(self, name, version, created, tags, changes):
        entry = {
            "name": name,
            "version": version,
            "created": created,
            "tags": tags,
            "changes": changes
        }
        history = []
        if self.prompt_history_file.exists():
            with open(self.prompt_history_file, 'r') as f:
                try:
                    history = json.load(f)
                except Exception:
                    history = []
        history.append(entry)
        with open(self.prompt_history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _log_experiment_history(self, experiment_id, config, timestamp, result_summary):
        entry = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "config": config,
            "result_summary": result_summary
        }
        history = []
        if self.experiment_history_file.exists():
            with open(self.experiment_history_file, 'r') as f:
                try:
                    history = json.load(f)
                except Exception:
                    history = []
        history.append(entry)
        with open(self.experiment_history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    # Prompt Management
    def save_prompt(self, name: str, prompt: str, version: Optional[str] = None, 
                   tags: Optional[List[str]] = None, changes: Optional[str] = None) -> str:
        """Save a prompt with versioning."""
        prompt_dir = self.prompts_dir / name
        prompt_dir.mkdir(exist_ok=True)
        
        # Determine version
        if version is None:
            existing_versions = self._get_prompt_versions(name)
            if existing_versions:
                latest = max(existing_versions, key=lambda v: int(v.replace('v', '')))
                version_num = int(latest.replace('v', '')) + 1
                version = f"v{version_num}"
            else:
                version = "v1"
        
        # Create prompt version
        prompt_version = PromptVersion(
            prompt=prompt,
            version=version,
            created=datetime.now().isoformat(),
            parent=self._get_latest_prompt_version(name),
            changes=changes,
            tags=tags or []
        )
        
        # Save version file
        version_file = prompt_dir / f"{version}.yaml"
        with open(version_file, 'w') as f:
            yaml.dump(asdict(prompt_version), f, default_flow_style=False)
        
        # Update metadata
        self._update_prompt_metadata(name, prompt_version)
        # Log to prompt history
        self._log_prompt_history(name, version, prompt_version.created, tags or [], changes)
        
        return version
    
    def get_prompt(self, name: str, version: Optional[str] = None) -> Optional[str]:
        """Get a prompt by name and version."""
        if version is None:
            version = self._get_latest_prompt_version(name)
        
        if not version:
            return None
        
        version_file = self.prompts_dir / name / f"{version}.yaml"
        if not version_file.exists():
            return None
        
        with open(version_file, 'r') as f:
            data = yaml.safe_load(f)
            return data['prompt']
    
    def list_prompts(self, tags_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List all prompts with optional tag filtering."""
        prompts = []
        
        for prompt_dir in self.prompts_dir.iterdir():
            if not prompt_dir.is_dir():
                continue
            
            name = prompt_dir.name
            metadata_file = prompt_dir / "metadata.yaml"
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = yaml.safe_load(f)
                
                # Apply tag filter
                if tags_filter:
                    prompt_tags = metadata.get('tags', [])
                    if not any(tag in prompt_tags for tag in tags_filter):
                        continue
                
                prompts.append({
                    'name': name,
                    'description': metadata.get('description', ''),
                    'latest_version': metadata.get('latest_version', ''),
                    'tags': metadata.get('tags', []),
                    'created': metadata.get('created', '')
                })
        
        return sorted(prompts, key=lambda p: p['name'])
    
    def delete_prompt(self, name: str) -> bool:
        """Delete a prompt and all its versions."""
        prompt_dir = self.prompts_dir / name
        if not prompt_dir.exists():
            return False
        
        import shutil
        shutil.rmtree(prompt_dir)
        return True
    
    def get_prompt_versions(self, name: str) -> List[Dict[str, Any]]:
        """Get all versions of a prompt."""
        prompt_dir = self.prompts_dir / name
        if not prompt_dir.exists():
            return []
        
        versions = []
        for version_file in prompt_dir.glob("v*.yaml"):
            if version_file.name == "metadata.yaml":
                continue
            
            with open(version_file, 'r') as f:
                data = yaml.safe_load(f)
                versions.append({
                    'version': data['version'],
                    'created': data['created'],
                    'changes': data.get('changes', ''),
                    'tags': data.get('tags', [])
                })
        # Sort numerically by version (e.g., v2 < v10)
        def version_key(v):
            try:
                return int(v['version'].replace('v', ''))
            except Exception:
                return 0
        return sorted(versions, key=version_key)
    
    def diff_prompts(self, name1: str, name2: str, version1: Optional[str] = None, 
                    version2: Optional[str] = None) -> Optional[str]:
        """Get diff between two prompts."""
        prompt1 = self.get_prompt(name1, version1)
        prompt2 = self.get_prompt(name2, version2)
        
        if prompt1 is None or prompt2 is None:
            # Debug output for diagnosis
            print(f"[DEBUG] diff_prompts: prompt1={prompt1}, prompt2={prompt2}")
            print(f"[DEBUG] name1={name1}, version1={version1}, name2={name2}, version2={version2}")
            prompt1_path = self.prompts_dir / name1 / f"{version1}.yaml" if version1 else None
            prompt2_path = self.prompts_dir / name2 / f"{version2}.yaml" if version2 else None
            print(f"[DEBUG] prompt1_path={prompt1_path}, prompt2_path={prompt2_path}")
            return None
        
        import difflib
        diff = difflib.unified_diff(
            prompt1.splitlines(),
            prompt2.splitlines(),
            fromfile=f"{name1}{'/' + version1 if version1 else ''}",
            tofile=f"{name2}{'/' + version2 if version2 else ''}",
            lineterm=''
        )
        diff_lines = list(diff)
        if not diff_lines:
            return None
        color_diff = []
        for line in diff_lines:
            if line.startswith('+') and not line.startswith('+++'):
                color_diff.append(f"\033[92m{line}\033[0m")  # Green
            elif line.startswith('-') and not line.startswith('---'):
                color_diff.append(f"\033[91m{line}\033[0m")  # Red
            else:
                color_diff.append(line)
        return '\n'.join(color_diff)
    
    # Metric Management
    def save_metric(self, name: str, metric_type: str, description: str, 
                   implementation: str, parameters: Dict[str, Any]) -> bool:
        """Save a metric definition."""
        metric = Metric(
            name=name,
            type=metric_type,
            description=description,
            implementation=implementation,
            parameters=parameters
        )
        
        metric_file = self.metrics_dir / f"{name}.yaml"
        with open(metric_file, 'w') as f:
            yaml.dump(asdict(metric), f, default_flow_style=False)
        
        return True
    
    def list_metrics(self) -> List[Dict[str, Any]]:
        """List all available metrics."""
        metrics = []
        
        for metric_file in self.metrics_dir.glob("*.yaml"):
            with open(metric_file, 'r') as f:
                data = yaml.safe_load(f)
                metrics.append({
                    'name': data['name'],
                    'type': data['type'],
                    'description': data['description']
                })
        
        return sorted(metrics, key=lambda m: m['name'])
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        metric_file = self.metrics_dir / f"{name}.yaml"
        if not metric_file.exists():
            return None
        
        with open(metric_file, 'r') as f:
            data = yaml.safe_load(f)
            return Metric(**data)
    
    # Experiment Management
    def save_experiment(self, experiment: Experiment) -> bool:
        """Save an experiment configuration."""
        experiment_dir = self.experiments_dir / experiment.experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        config_file = experiment_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(asdict(experiment), f, default_flow_style=False)
        
        return True
    
    def list_experiments(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all experiments."""
        experiments = []
        
        for experiment_dir in self.experiments_dir.iterdir():
            if not experiment_dir.is_dir():
                continue
            
            config_file = experiment_dir / "config.yaml"
            if not config_file.exists():
                continue
            
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)
            
            # Check for results
            results_file = experiment_dir / "results.json"
            status = "completed" if results_file.exists() else "pending"
            
            if status_filter and status != status_filter:
                continue
            
            experiments.append({
                'experiment_id': data['experiment_id'],
                'name': data['name'],
                'created': data['created'],
                'status': status
            })
        
        return sorted(experiments, key=lambda e: e['created'], reverse=True)
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        config_file = self.experiments_dir / experiment_id / "config.yaml"
        if not config_file.exists():
            return None
        
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)
            return Experiment(**data)
    
    def save_experiment_results(self, experiment_id: str, results: List[Dict[str, Any]], config: dict = None) -> bool:
        """Save experiment results."""
        experiment_dir = self.experiments_dir / experiment_id
        experiment_dir.mkdir(exist_ok=True)
        results_file = experiment_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        # Log to experiment history
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        # Summarize results for history
        summary = {
            "cases": len(results),
            "A_wins": sum(1 for r in results if r.get('winner') == 'A'),
            "B_wins": sum(1 for r in results if r.get('winner') == 'B'),
            "Ties": sum(1 for r in results if r.get('winner') == 'Tie')
        }
        self._log_experiment_history(experiment_id, config or {}, timestamp, summary)
        return True
    
    # Helper methods
    def _get_prompt_versions(self, name: str) -> List[str]:
        """Get all version numbers for a prompt."""
        prompt_dir = self.prompts_dir / name
        if not prompt_dir.exists():
            return []
        
        versions = []
        for version_file in prompt_dir.glob("v*.yaml"):
            if version_file.name == "metadata.yaml":
                continue
            versions.append(version_file.stem)
        
        return versions
    
    def _get_latest_prompt_version(self, name: str) -> Optional[str]:
        """Get the latest version of a prompt."""
        versions = self._get_prompt_versions(name)
        if not versions:
            return None
        
        return max(versions, key=lambda v: int(v.replace('v', '')))
    
    def _update_prompt_metadata(self, name: str, prompt_version: PromptVersion):
        """Update prompt metadata."""
        metadata_file = self.prompts_dir / name / "metadata.yaml"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = yaml.safe_load(f)
        else:
            metadata = {
                'name': name,
                'description': '',
                'category': 'general',
                'author': 'unknown',
                'created': prompt_version.created,
                'tags': []
            }
        
        metadata['latest_version'] = prompt_version.version
        if prompt_version.tags:
            metadata['tags'] = list(set(metadata.get('tags', []) + prompt_version.tags))
        
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False) 