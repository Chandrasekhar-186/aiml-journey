# Databricks Phone Screen Problem
# IP Firewall with CIDR matching
# Date: May 27, 2026

from typing import List, Tuple

class IpFirewall:
    """
    IP Firewall with CIDR block matching.
    First matching rule determines result.
    """
    def __init__(self,
                  rules: List[Tuple[str, str]]):
        self.rules = []
        for action, ip_or_cidr in rules:
            self.rules.append(
                (action,
                 *self._parse_cidr(ip_or_cidr))
            )

    def _ip_to_int(self, ip: str) -> int:
        """Convert "192.168.1.1" → integer"""
        parts = ip.split('.')
        result = 0
        for part in parts:
            result = (result << 8) | int(part)
        return result

    def _parse_cidr(self, cidr: str):
        """Parse "192.168.1.0/24" → (network, mask)"""
        if '/' in cidr:
            ip, prefix = cidr.split('/')
            prefix = int(prefix)
        else:
            ip = cidr
            prefix = 32  # single IP = /32

        network = self._ip_to_int(ip)
        # Mask: first 'prefix' bits are 1s
        if prefix == 0:
            mask = 0
        else:
            mask = ((1 << 32) - 1) - \
                   ((1 << (32 - prefix)) - 1)

        # Normalize network address
        network = network & mask
        return network, mask

    def query(self, ip: str) -> str:
        """Find first matching rule"""
        ip_int = self._ip_to_int(ip)

        for action, network, mask in self.rules:
            if ip_int & mask == network:
                return action

        return "DENY"  # default deny!

# Test cases
rules = [
    ("ALLOW", "192.168.100.5/30"),
    ("DENY",  "10.0.0.0/8"),
    ("ALLOW", "1.2.3.4")
]
fw = IpFirewall(rules)

# /30 = last 2 bits free → 192.168.100.4-7
print(fw.query("192.168.100.4"))  # ALLOW ✅
print(fw.query("192.168.100.7"))  # ALLOW ✅
print(fw.query("192.168.100.8"))  # DENY (no match)
print(fw.query("10.5.6.7"))       # DENY ✅
print(fw.query("1.2.3.4"))        # ALLOW ✅
print(fw.query("8.8.8.8"))        # DENY (default)

print("\nKey concepts:")
print("""
CIDR /24 = first 24 bits fixed
         = last 8 bits free (256 IPs)

IP matching: (ip & mask) == (network & mask)

Mask for /24:
11111111.11111111.11111111.00000000
= 0xFFFFFF00

mask = ((1<<32)-1) - ((1<<(32-prefix))-1)

First matching rule wins!
Default action = DENY (security best practice)
""")
