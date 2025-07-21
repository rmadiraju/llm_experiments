# User Management System - Spring Boot Application

## Project Overview
Create a comprehensive User Management System using Java Spring Boot with the following features:

## Functional Requirements

### 1. User Management
- **User Registration**: Allow users to register with email, password, and basic information
- **User Authentication**: Implement JWT-based authentication
- **User Profile**: Users can view and update their profile information
- **User Roles**: Support different user roles (ADMIN, USER, MODERATOR)
- **Password Reset**: Implement password reset functionality via email

### 2. User Profile Features
- **Profile Information**: Store user details (name, email, phone, address, profile picture)
- **Profile Updates**: Allow users to update their information
- **Profile Picture**: Support profile picture upload and storage

### 3. Security Features
- **JWT Authentication**: Secure API endpoints with JWT tokens
- **Password Encryption**: Encrypt passwords using BCrypt
- **Role-based Access Control**: Implement authorization based on user roles
- **Input Validation**: Validate all user inputs
- **Rate Limiting**: Implement rate limiting for API endpoints

### 4. API Endpoints
- **Authentication**:
  - POST `/api/auth/register` - User registration
  - POST `/api/auth/login` - User login
  - POST `/api/auth/logout` - User logout
  - POST `/api/auth/refresh` - Refresh JWT token
  - POST `/api/auth/forgot-password` - Request password reset
  - POST `/api/auth/reset-password` - Reset password

- **User Management**:
  - GET `/api/users` - Get all users (admin only)
  - GET `/api/users/{id}` - Get user by ID
  - GET `/api/users/profile` - Get current user profile
  - PUT `/api/users/profile` - Update current user profile
  - DELETE `/api/users/{id}` - Delete user (admin only)
  - PUT `/api/users/{id}/role` - Update user role (admin only)

- **File Upload**:
  - POST `/api/files/upload` - Upload profile picture
  - GET `/api/files/{filename}` - Get uploaded file

## Technical Requirements

### 1. Technology Stack
- **Framework**: Spring Boot 3.x
- **Java Version**: Java 17 or higher
- **Database**: PostgreSQL
- **Build Tool**: Maven
- **Security**: Spring Security with JWT
- **Documentation**: OpenAPI 3 (Swagger)

### 2. Project Structure
```
src/main/java/com/usermanagement/
├── config/
│   ├── SecurityConfig.java
│   ├── JwtConfig.java
│   └── WebConfig.java
├── controller/
│   ├── AuthController.java
│   ├── UserController.java
│   └── FileController.java
├── service/
│   ├── AuthService.java
│   ├── UserService.java
│   ├── EmailService.java
│   └── FileService.java
├── repository/
│   ├── UserRepository.java
│   └── RoleRepository.java
├── entity/
│   ├── User.java
│   ├── Role.java
│   └── RefreshToken.java
├── dto/
│   ├── UserDto.java
│   ├── LoginDto.java
│   ├── RegisterDto.java
│   └── PasswordResetDto.java
├── exception/
│   ├── GlobalExceptionHandler.java
│   └── CustomExceptions.java
└── util/
    ├── JwtUtil.java
    └── PasswordUtil.java
```

### 3. Database Schema
- **users**: id, email, password, first_name, last_name, phone, address, profile_picture, created_at, updated_at, enabled
- **roles**: id, name, description
- **user_roles**: user_id, role_id
- **refresh_tokens**: id, user_id, token, expiry_date

### 4. Configuration Files
- **application.yml**: Main configuration
- **application-dev.yml**: Development environment
- **application-prod.yml**: Production environment
- **pom.xml**: Maven dependencies

## Non-Functional Requirements

### 1. Performance
- API response time < 200ms for most operations
- Support for 1000+ concurrent users
- Efficient database queries with proper indexing

### 2. Security
- HTTPS in production
- Input sanitization
- SQL injection prevention
- XSS protection
- CORS configuration

### 3. Logging and Monitoring
- Structured logging with SLF4J
- Request/response logging
- Error tracking and alerting
- Performance metrics

### 4. Testing
- Unit tests for all services
- Integration tests for controllers
- API tests for all endpoints
- Test coverage > 80%

## Deployment Requirements

### 1. Containerization
- Docker support with Dockerfile
- Docker Compose for local development
- Multi-stage Docker builds

### 2. Environment Configuration
- Environment-specific configurations
- Externalized configuration
- Secrets management

### 3. Documentation
- Comprehensive README.md
- API documentation with Swagger
- Setup and deployment guides
- Code documentation with JavaDoc

## Additional Features

### 1. Email Integration
- Email verification for new registrations
- Password reset emails
- Welcome emails for new users

### 2. File Storage
- Local file storage for development
- Cloud storage support (AWS S3) for production
- Image compression and optimization

### 3. Audit Trail
- Track user actions and changes
- Log important events
- Maintain audit history

## Success Criteria
1. All API endpoints working correctly
2. Proper authentication and authorization
3. Database operations working efficiently
4. Comprehensive test coverage
5. Clean, maintainable code structure
6. Complete documentation
7. Docker containerization working
8. Security best practices implemented 