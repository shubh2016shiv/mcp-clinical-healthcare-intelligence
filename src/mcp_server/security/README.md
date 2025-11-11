# Healthcare MCP Server Security Layer

## Overview

This security layer implements enterprise-grade security controls for the healthcare MCP server, specifically designed to protect Protected Health Information (PHI) and FHIR data stored in MongoDB. The implementation follows zero-trust principles and ensures HIPAA compliance.

## Security Principles Applied

### 1. Defense in Depth
Multiple layers of security controls working together:
- **Authentication**: Verifies user identity
- **Authorization**: Enforces access permissions
- **Input Validation**: Prevents injection attacks
- **Audit Logging**: Records all access attempts
- **Data Minimization**: Limits data exposure
- **Rate Limiting**: Prevents abuse

### 2. Least Privilege
Users receive only the minimum permissions necessary for their role:
- **ADMIN**: Full read access to all data (system administration)
- **CLINICIAN**: Complete read access to clinical data for patient care
- **RESEARCHER**: Read access to aggregated/de-identified data only
- **BILLING**: Read access to financial data only
- **READ_ONLY**: Limited read access with minimal fields

**Security Best Practice**: MCP servers are read-only by design. No write
operations are permitted through MCP tools. This reduces attack surface,
ensures data integrity, and follows the principle of least privilege. All
data modifications must occur through separate, secured APIs with proper
authorization.

### 3. Zero Trust
Every request is validated, trust is never assumed:
- All requests require authentication
- Session validation on every request
- IP address verification to detect hijacking
- Permission checking at multiple levels

### 4. Audit Everything
Comprehensive logging of all PHI access (HIPAA Security Rule §164.312(b)):
- Who accessed data (user_id, role, IP)
- What was accessed (resource type, IDs)
- When accessed (timestamp)
- Why accessed (operation/purpose)
- Result (success/failure, records returned)

### 5. Fail Secure
Safe defaults that prevent data exposure:
- Deny access by default
- Explicit allow through permissions
- Secure error messages (no data leakage)
- Graceful degradation on failures

## HIPAA Compliance Features

### Security Rule Requirements
- **Access Control**: RBAC with read-only role-based permissions
- **Audit Trails**: 7-year retention of access logs
- **Integrity**: Data validation and sanitization (read-only operations)
- **Authentication**: Multi-factor ready session management
- **Transmission Security**: TLS enforcement for MongoDB
- **Read-Only Design**: No write operations reduce attack surface

### Privacy Rule Requirements
- **Minimum Necessary**: Data minimization by role
- **De-identification**: Safe Harbor methods for research
- **Access Tracking**: Complete audit of PHI access
- **Breach Prevention**: Multiple security layers

## Architecture

```
src/mcp_server/security/
├── __init__.py              # Security module exports and SecurityManager
├── config.py                # SecurityConfig with HIPAA-compliant defaults
├── authentication.py        # UserRole, SecurityContext, AuthenticationManager
├── authorization.py         # @require_auth decorator and permissions
├── validation.py            # InputValidator for NoSQL injection prevention
├── audit.py                 # AuditLogger for HIPAA audit trail
├── data_minimization.py     # DataMinimizer for minimum necessary principle
├── middleware.py            # FastMCP integration for security context
└── README.md               # This documentation
```

## Read-Only Design Principle

**Critical Security Best Practice**: This MCP server is read-only by design.
This is a fundamental security principle that provides multiple benefits:

### Why Read-Only?
1. **Reduced Attack Surface**: No write operations means fewer attack vectors
2. **Data Integrity**: Prevents accidental or malicious data modification
3. **Separation of Concerns**: Data modifications handled by separate, secured APIs
4. **Compliance**: Easier to audit and demonstrate compliance (only reads logged)
5. **Least Privilege**: Users only get read access, never write access

### Implementation
- All MCP tools use read-only permissions (`read_phi`, `read_financial`, etc.)
- No write permissions exist in the permission system
- Database queries are read-only (find, aggregate, count operations only)
- All roles have read-only access regardless of privilege level

### Data Modifications
If data modifications are required, they must be handled through:
- Separate REST APIs with proper authorization
- Direct database access with appropriate security controls
- ETL pipelines with audit trails
- Administrative interfaces with multi-factor authentication

**Never** implement write operations in MCP tools. This violates security
best practices and increases the attack surface unnecessarily.

## Key Components

### SecurityConfig
Centralized configuration with HIPAA-compliant defaults:
- Rate limits: 60 requests/minute
- Query limits: 100 max results
- Session timeout: 30 minutes
- Audit retention: 7 years (2555 days)
- Field whitelisting for query validation

### AuthenticationManager
Handles user authentication and session management:
- Secure session token generation
- IP address validation (anti-hijacking)
- Rate limiting per client
- Failed attempt tracking with lockout

### InputValidator
Prevents NoSQL injection and validates inputs:
- MongoDB operator injection prevention
- Field whitelisting
- String sanitization
- Query depth validation
- Patient ID format validation

### AuditLogger
HIPAA-compliant audit logging:
- Structured JSON audit logs
- PII redaction options
- Separate audit file handler
- Comprehensive access tracking

### DataMinimizer
Implements minimum necessary principle:
- Role-based field filtering
- De-identification for research
- Safe Harbor compliant methods
- Age group categorization

### Authorization Decorator
Enforces permissions on MCP tools:
- Declarative security with decorators
- Automatic audit logging
- Permission checking per operation
- Integration with FastMCP middleware

## Usage Examples

### Tool Security Integration
```python
from src.mcp_server.security import require_auth

@server.tool()
@require_auth("read_phi")
async def search_patients(first_name: str = None) -> Dict[str, Any]:
    # Tool automatically gets security context and audit logging
    security_context = get_current_security_context()
    # Implementation with automatic data minimization
    pass
```

### Manual Security Checking
```python
from src.mcp_server.security import get_security_manager

security_manager = get_security_manager()

# Validate request
if security_manager.validate_request(context, "search_patients", query_params):
    # Process request
    results = security_manager.data_minimizer.filter_fields(data, context.role)
    pass
```

### Configuration
```python
from src.config.settings import settings

# Security settings via environment variables
export SECURITY_ENABLED=true
export SECURITY_DEFAULT_ROLE=read_only
export SECURITY_SESSION_TIMEOUT=30
export SECURITY_RATE_LIMIT=60
export SECURITY_MAX_QUERY_RESULTS=100
export SECURITY_AUDIT_RETENTION_DAYS=2555
```

## Security Measures Rationale

### 1. NoSQL Injection Prevention
**Threat**: MongoDB operators ($where, $regex, $expr) can bypass access controls
**Defense**: Whitelist validation, operator detection, query depth limits
**HIPAA Impact**: Prevents unauthorized data access and potential breaches

### 2. Rate Limiting
**Threat**: Brute force attacks, DoS attacks, bulk data extraction
**Defense**: Per-client sliding window algorithm with configurable limits
**HIPAA Impact**: Protects system availability and prevents abuse

### 3. Session Security
**Threat**: Session hijacking, stale authenticated sessions
**Defense**: IP validation, timeout enforcement, secure token generation
**HIPAA Impact**: Ensures authenticated users remain authenticated

### 4. Data Minimization
**Threat**: Excessive PHI exposure, privacy violations
**Defense**: Role-based field filtering, de-identification methods
**HIPAA Impact**: Complies with minimum necessary principle

### 5. Audit Logging
**Threat**: Undetected unauthorized access, compliance violations
**Defense**: Comprehensive logging with 7-year retention
**HIPAA Impact**: Required by Security Rule, enables breach investigation

### 6. Input Validation
**Threat**: Malformed queries, injection attacks, DoS via large inputs
**Defense**: Type validation, length limits, pattern matching
**HIPAA Impact**: Ensures data integrity and system stability

## Deployment Checklist

### Pre-Deployment Security Review
- [ ] MongoDB authentication enabled (`--auth` flag)
- [ ] TLS/SSL enabled for MongoDB connections
- [ ] Strong database passwords (32+ characters)
- [ ] Network isolation (MongoDB binds to private IP only)
- [ ] Encryption at rest enabled
- [ ] Regular backup procedures with encryption

### Configuration Validation
- [ ] `SECURITY_ENABLED=true` in environment
- [ ] Audit log path exists with proper permissions
- [ ] Security settings validated via `settings.validate_configuration()`
- [ ] HIPAA compliance check passes (`security_manager.is_hipaa_compliant()`)

### Runtime Security
- [ ] Security layer initializes without errors
- [ ] Audit logging functional (check audit.log file)
- [ ] Tool permissions correctly enforced
- [ ] Data minimization working by role
- [ ] Rate limiting active and configurable

### Monitoring & Alerting
- [ ] Security events logged to monitoring system
- [ ] Failed authentication attempts tracked
- [ ] High-volume queries flagged
- [ ] Regular audit log review process established

## Compliance Evidence

### HIPAA Security Rule (§164.312)
- **Access Control**: RBAC implemented with `@require_auth` decorator
- **Audit Trails**: `AuditLogger` provides comprehensive access logging
- **Integrity**: `InputValidator` ensures data integrity
- **Authentication**: Session management with IP validation
- **Transmission Security**: TLS enforcement configurable

### HIPAA Privacy Rule (§164.502)
- **Minimum Necessary**: `DataMinimizer` filters data by role
- **De-identification**: Safe Harbor methods implemented
- **Access Tracking**: Complete audit trail maintained
- **Breach Prevention**: Multiple security layers

## Testing Security

### Unit Tests
```bash
# Test individual security components
pytest src/mcp_server/security/ -v

# Test specific security features
pytest tests/security/test_input_validation.py
pytest tests/security/test_audit_logging.py
pytest tests/security/test_data_minimization.py
```

### Integration Tests
```bash
# Test tool security integration
pytest tests/integration/test_secure_tools.py

# Test HIPAA compliance
pytest tests/compliance/test_hipaa_requirements.py
```

### Security Testing
```bash
# Test injection prevention
pytest tests/security/test_injection_attacks.py

# Test rate limiting
pytest tests/security/test_rate_limiting.py

# Test audit completeness
pytest tests/security/test_audit_completeness.py
```

## Troubleshooting

### Common Issues

**Security initialization fails**
- Check environment variables are set correctly
- Verify audit log directory is writable
- Ensure MongoDB connection is available

**Permission denied errors**
- Verify user roles are correctly assigned
- Check `@require_auth` decorators on tools
- Validate security context is properly set

**Audit logs not writing**
- Check file permissions on audit.log path
- Verify disk space is available
- Ensure logger configuration is correct

**Rate limiting too aggressive**
- Adjust `SECURITY_RATE_LIMIT` environment variable
- Check client IP detection is working
- Review legitimate usage patterns

### Debug Mode
Enable verbose security logging:
```bash
export LOG_LEVEL=DEBUG
export SECURITY_DEBUG=true
```

This provides detailed security event logging for troubleshooting.

## Future Enhancements

### Advanced Features
- Multi-factor authentication (MFA) support
- OAuth 2.0 / OpenID Connect integration
- Real-time security monitoring dashboard
- Automated threat detection
- Integration with SIEM systems

### Compliance Improvements
- Automated compliance reporting
- Integration with compliance management platforms
- Advanced de-identification techniques
- Privacy impact assessments

### Performance Optimizations
- Caching for permission checks
- Asynchronous audit logging
- Optimized query validation
- Connection pooling for security services

## References

- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/)
- [HIPAA Privacy Rule](https://www.hhs.gov/hipaa/for-professionals/privacy/)
- [Safe Harbor De-identification](https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [OWASP API Security](https://owasp.org/www-project-api-security/)

---

**Security Notice**: This security layer is designed for healthcare environments handling PHI. Always consult with legal and compliance experts before deploying in production. Regular security audits and penetration testing are recommended.
