from dependency_manager import DependencyManager
import tempfile

dm = DependencyManager(cache_dir=tempfile.mkdtemp(), trust_mode='safe')
print('Trusted sources:', dm.trusted_sources)

domain = 'untrusted.com'
print('Domain:', domain)

for trusted in dm.trusted_sources:
    print(f'Checking {trusted}:')
    print(f'  domain == trusted: {domain == trusted}')
    print(f'  domain.endswith("."+trusted): {domain.endswith("." + trusted)}')
    print(f'  trusted.startswith(domain+"/"): {trusted.startswith(domain + "/")}')
    if domain == trusted or domain.endswith('.' + trusted) or trusted.startswith(domain + '/'):
        print(f'  MATCH FOUND with {trusted}')
        break
else:
    print('  No matches found')

result = dm._validate_source_security("untrusted.com/model")
print('Result:', result)
print('is_trusted should be False:', not any(
    domain == trusted or domain.endswith('.' + trusted) or trusted.startswith(domain + '/')
    for trusted in dm.trusted_sources
))