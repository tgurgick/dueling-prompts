#!/usr/bin/env python3
"""
Duel - Unified prompt management and testing tool

Usage:
    duel init                    # Initialize duel workspace
    duel save <name> <prompt>    # Save a prompt
    duel list                    # List saved prompts
    duel test <prompt1> <prompt2> [options]  # Test two prompts
    duel run <config.yaml>       # Run from YAML config
"""

import yaml
import sys
import os
import tempfile
import subprocess
import argparse
from pathlib import Path
import typing
import traceback

class DuelManager:
    def __init__(self):
        self.duel_dir = Path(".duel")
        self.prompts_file = self.duel_dir / "prompts.yaml"
        self.duel_dir.mkdir(exist_ok=True)
    
    def init(self):
        """Initialize the duel workspace."""
        if not self.prompts_file.exists():
            self.prompts_file.write_text("prompts: {}\n")
            print("✅ Duel workspace initialized!")
        else:
            print("✅ Duel workspace already exists!")
    
    def save_prompt(self, name, prompt):
        """Save a prompt with the given name."""
        prompts = self._load_prompts()
        prompts[name] = prompt
        self._save_prompts(prompts)
        print(f"✅ Saved prompt '{name}': {prompt[:50]}...")
    
    def list_prompts(self):
        """List all saved prompts."""
        prompts = self._load_prompts()
        if not prompts:
            print("📝 No prompts saved yet.")
            print("💡 Use 'duel save <name> <prompt>' to save prompts")
            return
        
        print("📝 Saved prompts:")
        for name, prompt in prompts.items():
            print(f"  {name}: {prompt[:60]}...")
    
    def get_prompt(self, name):
        """Get a saved prompt by name."""
        prompts = self._load_prompts()
        return prompts.get(name)
    
    def delete_prompt(self, name):
        """Delete a saved prompt by name."""
        prompts = self._load_prompts()
        if name not in prompts:
            print(f"❌ Prompt '{name}' not found.")
            print("💡 Use 'duel list' to see available prompts")
            return False
        
        deleted_prompt = prompts.pop(name)
        self._save_prompts(prompts)
        print(f"🗑️  Deleted prompt '{name}': {deleted_prompt[:50]}...")
        return True
    
    def diff_prompts(self, prompt1, prompt2):
        """Show differences between two prompts."""
        # Resolve prompt names if they're saved prompts
        prompt_a = self.get_prompt(prompt1) or prompt1
        prompt_b = self.get_prompt(prompt2) or prompt2
        
        print(f"🔍 Comparing prompts:")
        print(f"  A: {prompt1}")
        print(f"  B: {prompt2}")
        print()
        
        # Simple character-by-character diff with highlighting
        self._show_diff(prompt_a, prompt_b)
    
    def _show_diff(self, text_a, text_b):
        """Show a simple diff between two texts."""
        import difflib
        
        # Split into lines for better diff visualization
        lines_a = text_a.splitlines()
        lines_b = text_b.splitlines()
        
        # Generate diff
        diff = difflib.unified_diff(
            lines_a, lines_b,
            fromfile='Prompt A',
            tofile='Prompt B',
            lineterm=''
        )
        
        # Print diff with color coding
        diff_lines = list(diff)
        
        if not diff_lines:
            print("✅ Prompts are identical!")
            return
        
        print("📝 Differences:")
        print("-" * 50)
        
        for line in diff_lines:
            if line.startswith('---') or line.startswith('+++'):
                # Skip diff headers
                continue
            elif line.startswith('@@'):
                # Show context
                print(f"  {line}")
            elif line.startswith('+'):
                # Added line (green)
                print(f"  \033[92m{line}\033[0m")  # Green
            elif line.startswith('-'):
                # Removed line (red)
                print(f"  \033[91m{line}\033[0m")  # Red
            else:
                # Context line
                print(f"  {line}")
        
        print("-" * 50)
        
        # Summary
        added_lines = sum(1 for line in diff_lines if line.startswith('+') and not line.startswith('+++'))
        removed_lines = sum(1 for line in diff_lines if line.startswith('-') and not line.startswith('---'))
        
        if added_lines > 0 or removed_lines > 0:
            print(f"📊 Summary: {removed_lines} lines removed, {added_lines} lines added")
        else:
            print("📊 Summary: No differences found")
    
    def test_prompts(self, prompt1, prompt2, inputs, 
                    expected=None, model="gpt-4o-mini", 
                    metric="exact", save_csv=False):
        """Test two prompts against inputs."""
        
        # Resolve prompt names if they're saved prompts
        prompt_a = self.get_prompt(prompt1) or prompt1
        prompt_b = self.get_prompt(prompt2) or prompt2
        
        print(f"🏆 Testing prompts:")
        print(f"  A: {prompt_a}")
        print(f"  B: {prompt_b}")
        print(f"📊 Inputs: {len(inputs)}")
        print()
        
        # Create temporary config
        config = {
            "experiment": f"Quick Test: {prompt1} vs {prompt2}",
            "model": model,
            "prompts": {"A": prompt_a, "B": prompt_b},
            "cases": [],
            "metric": metric
        }
        
        for i, input_text in enumerate(inputs):
            case = {"input": input_text}
            if expected and i < len(expected):
                case["expected"] = expected[i]
            config["cases"].append(case)
        
        # Write to temporary file and run
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config, temp_file, default_flow_style=False)
        temp_file.close()
        
        try:
            # Import and run the core engine
            from .engine import PromptDuel
            duel = PromptDuel(temp_file.name)
            results = duel.run_duel()
            duel.print_results(results)
            
            if save_csv:
                duel.save_csv(results)
            
        finally:
            os.unlink(temp_file.name)
    
    def run_config(self, config_file, save_csv=False):
        """Run a duel from a YAML config file."""
        from .engine import PromptDuel
        duel = PromptDuel(config_file)
        results = duel.run_duel()
        duel.print_results(results)
        
        if save_csv:
            duel.save_csv(results)
    
    def _load_prompts(self):
        """Load saved prompts from YAML file."""
        if not self.prompts_file.exists():
            return {}
        
        try:
            with open(self.prompts_file, 'r') as f:
                data = yaml.safe_load(f)
                return data.get('prompts', {})
        except Exception as e:
            print(f"❌ Error loading prompts: {e}")
            return {}
    
    def _save_prompts(self, prompts):
        """Save prompts to YAML file."""
        data = {"prompts": prompts}
        with open(self.prompts_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser(
        description="Duel - Unified prompt management and testing tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  duel init                                    # Initialize workspace
  duel save summarize "Summarize: {input}"     # Save a prompt
  duel save tldr "TL;DR: {input}"              # Save another prompt
  duel list                                    # List saved prompts
  duel delete summarize                        # Delete a prompt
  duel diff summarize tldr                     # Show differences between prompts
  duel test summarize tldr -i "Hello world"    # Test two prompts
  duel run config.yaml                         # Run from config file
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    subparsers.add_parser('init', help='Initialize duel workspace')
    
    # Save command
    save_parser = subparsers.add_parser('save', help='Save a prompt')
    save_parser.add_argument('name', help='Prompt name')
    save_parser.add_argument('prompt', help='Prompt template (use {input} for test input)')
    
    # List command
    subparsers.add_parser('list', help='List saved prompts')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a saved prompt')
    delete_parser.add_argument('name', help='Name of the prompt to delete')
    
    # Diff command
    diff_parser = subparsers.add_parser('diff', help='Show differences between two prompts')
    diff_parser.add_argument('prompt1', help='First prompt (name or template)')
    diff_parser.add_argument('prompt2', help='Second prompt (name or template)')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test two prompts')
    test_parser.add_argument('prompt1', help='First prompt (name or template)')
    test_parser.add_argument('prompt2', help='Second prompt (name or template)')
    test_parser.add_argument('-i', '--inputs', nargs='+', required=True, help='Test inputs')
    test_parser.add_argument('-e', '--expected', nargs='+', help='Expected outputs')
    test_parser.add_argument('-m', '--model', default='gpt-4o-mini', help='Model to use')
    test_parser.add_argument('--metric', default='exact', choices=['exact', 'contains', 'judge'], help='Scoring metric')
    test_parser.add_argument('--save-csv', action='store_true', help='Save results to CSV')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run from YAML config')
    run_parser.add_argument('config', help='YAML config file')
    run_parser.add_argument('--save-csv', action='store_true', help='Save results to CSV')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = DuelManager()
    
    try:
        if args.command == 'init':
            manager.init()
        
        elif args.command == 'save':
            manager.save_prompt(args.name, args.prompt)
        
        elif args.command == 'list':
            manager.list_prompts()
        
        elif args.command == 'delete':
            manager.delete_prompt(args.name)
        
        elif args.command == 'diff':
            manager.diff_prompts(args.prompt1, args.prompt2)
        
        elif args.command == 'test':
            manager.test_prompts(
                prompt1=args.prompt1,
                prompt2=args.prompt2,
                inputs=args.inputs,
                expected=args.expected,
                model=args.model,
                metric=args.metric,
                save_csv=args.save_csv
            )
        
        elif args.command == 'run':
            manager.run_config(args.config, args.save_csv)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 