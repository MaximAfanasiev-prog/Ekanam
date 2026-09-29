# Team access by public IP

User authorized an IP-based trial. The isolated gateway is ready, but external
reachability is BLOCKED: direct external curl to 91.77.168.161:28443 timed out.
Do not describe the service as publicly reachable until external verification passes.

Server LAN address: 192.168.1.14; observed public egress IP: 91.77.168.161.
Existing shared ports 80/443 and their proxy configuration were not changed.

## Administrator handoff

Configure TCP forwarding from WAN 28443 to 192.168.1.14:28443 and permit that
port through the relevant perimeter firewall. Confirm the LAN lease/reservation
keeps 192.168.1.14 stable. No raw API port needs opening.
We do not have router access and cannot verify its NAT/firewall rules.

After forwarding, check from a different network that:
- https://91.77.168.161:28443/ responds with 401 without credentials;
- authenticated page, API and thumbnail requests succeed;
- an actual search/upload completes.
A timeout is not evidence that the password or model is wrong.

## Gateway

Compose project maxim-reid-ip, container maxim-reid-ip-gateway-1.
Pinned Caddy image, user 1009, read-only root, 128 MiB, 0.5 CPU, log rotation.
Only TCP 28443 is published. Caddy joins maxim-reid-yolo_default and proxies to
maxim-reid-yolo-api-1:8080. Basic authentication covers all paths.
Backend and thumbnails remain private. TLS uses Caddy's internal CA and
default_sni 91.77.168.161 for IP clients which omit SNI. Admin API and automatic
HTTP redirects are disabled. NET_BIND_SERVICE is retained because the Caddy
executable carries that capability, even though the exposed port is above 1024.

Runtime directory outside Git:
artifacts/street-falcon/ip-gateway/
- Caddyfile: route and bcrypt password hash, mode 0600.
- credentials.json: generated random team password, mode 0600.
- data/config: private Caddy state and CA keys, directories mode 0700.
- gateway.env: LCT_GATEWAY_DIR, mode 0600.

Never add these runtime files to Git or share CA private keys. Only the public CA
certificate may be distributed for explicit trust setup. Browsers will warn
until the internal CA is trusted; a publicly trusted certificate is not configured.
See https://caddyserver.com/docs/automatic-https for internal CA behavior.
Do not send team credentials over plaintext HTTP.

Start/update from the repository:
~~~sh
docker compose --env-file /home/projects/hackathon_2026_lunopopicks/hackathon_maxim/artifacts/street-falcon/ip-gateway/gateway.env -p maxim-reid-ip -f compose.ip.yml up -d
~~~
Stop only this gateway:
~~~sh
docker compose --env-file /home/projects/hackathon_2026_lunopopicks/hackathon_maxim/artifacts/street-falcon/ip-gateway/gateway.env -p maxim-reid-ip -f compose.ip.yml down
~~~
The existing SSH tunnel to backend port 27816 continues to work independently.

## Evidence / standards postflight

research-python, Engineering Standards 0.2.9; no new policy exceptions.
Caddy config validated. TLS chain and IP identity verified against the generated
CA over loopback and LAN address. Anonymous root/ready/metrics returned 401;
authenticated root/ready/metrics returned 200 on both addresses.
External direct probe timed out after 8 seconds: NAT/firewall still needs owner
action. No public availability, production acceptance or GitLab gate claim.
Only Compose and documentation enter Git; no credentials or dataset files.
