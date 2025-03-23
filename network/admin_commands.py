import logging
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from tabulate import tabulate
from core.constants import InteractionLevel
from core.interactive_utils import InteractiveSession, InteractiveConfig

logger = logging.getLogger(__name__)

@dataclass
class NetworkStats:
    total_peers: int
    active_peers: int
    banned_peers: int
    circuit_breaks: int
    rate_limits: int
    pending_consensus: int

class AdminCommands:
    def __init__(self, p2p_network):
        self.network = p2p_network
        self.session = None

    async def start_interactive_shell(self):
        """Start interactive admin shell"""
        self.session = InteractiveSession(
            level=InteractionLevel.ADMIN,
            config=InteractiveConfig(timeout=300, safe_mode=True)
        )

        async with self.session:
            while True:
                command = await self.session.prompt("\nAdmin> ")
                if command == "exit":
                    break
                await self._handle_command(command)

    async def _handle_command(self, command: str):
        """Handle admin commands"""
        try:
            cmd_parts = command.split()
            if not cmd_parts:
                return

            cmd = cmd_parts[0].lower()
            args = cmd_parts[1:]

            if cmd == "show":
                if not args:
                    print("Available show commands: peers, banned, reputation, consensus")
                    return
                await self._handle_show_command(args[0], args[1:])

            elif cmd == "ban":
                if len(args) < 1:
                    print("Usage: ban <peer_id> [reason]")
                    return
                reason = " ".join(args[1:]) if len(args) > 1 else "Manual ban"
                await self.network._ban_peer(args[0], reason)

            elif cmd == "unban":
                if len(args) != 1:
                    print("Usage: unban <peer_id>")
                    return
                self.network.banned_peers.discard(args[0])
                print(f"Unbanned peer {args[0]}")

            elif cmd == "reset-circuit":
                if len(args) != 0:
                    print("Usage: reset-circuit")
                    return
                await self.network.circuit_breaker.reset()
                print("Circuit breaker reset")

            elif cmd == "stats":
                await self._show_network_stats()

            else:
                print(f"Unknown command: {cmd}")

        except Exception as e:
            logger.error(f"Command error: {str(e)}")
            print(f"Error executing command: {str(e)}")

    async def _handle_show_command(self, subcmd: str, args: list):
        """Handle show subcommands"""
        if subcmd == "peers":
            self._show_peer_table(self.network.peers)

        elif subcmd == "banned":
            banned = [(peer, self.network.suspicious_activity.get(peer, []))
                     for peer in self.network.banned_peers]
            self._show_banned_table(banned)

        elif subcmd == "reputation":
            if args:
                peer_id = args[0]
                await self._show_peer_reputation(peer_id)
            else:
                await self._show_reputation_table()

        elif subcmd == "consensus":
            await self._show_consensus_status()

        else:
            print(f"Unknown show command: {subcmd}")

    def _show_peer_table(self, peers: set):
        """Display active peers in table format"""
        rows = []
        for peer in peers:
            status = "Active"
            latency = self.network.peer_latencies.get(peer, 0)
            reputation = self.network.reputation_manager.get_reputation(peer)
            rows.append([peer, status, f"{latency:.2f}ms", f"{reputation:.2f}"])

        print("\nPeer Status:")
        print(tabulate(rows, headers=["Peer ID", "Status", "Latency", "Reputation"]))

    async def _show_consensus_status(self):
        """Show current consensus proposals and votes"""
        if not self.network.pending_state_changes:
            print("No pending consensus decisions")
            return

        print("\nPending Consensus Proposals:")
        for prop_id, proposal in self.network.pending_state_changes.items():
            votes_for = sum(1 for v in proposal['votes'].values() if v)
            total_votes = len(proposal['votes'])
            print(f"\nProposal {prop_id}:")
            print(f"Type: {proposal['change_type']}")
            print(f"Change: {proposal['change']}")
            print(f"Votes: {votes_for}/{total_votes}")

    async def _show_network_stats(self):
        """Show current network statistics"""
        stats = NetworkStats(
            total_peers=len(self.network.peers),
            active_peers=len([p for p in self.network.peers 
                            if p not in self.network.banned_peers]),
            banned_peers=len(self.network.banned_peers),
            circuit_breaks=self.network.circuit_breaker.failure_count,
            rate_limits=self.network.rate_limiter.limit_count,
            pending_consensus=len(self.network.pending_state_changes)
        )

        print("\nNetwork Statistics:")
        print(f"Total Peers: {stats.total_peers}")
        print(f"Active Peers: {stats.active_peers}")
        print(f"Banned Peers: {stats.banned_peers}")
        print(f"Circuit Breaks: {stats.circuit_breaks}")
        print(f"Rate Limits: {stats.rate_limits}")
        print(f"Pending Consensus: {stats.pending_consensus}")
