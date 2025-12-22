//! Role-Based Access Control (RBAC) for package distribution
//!
//! This module provides fine-grained access control for package operations
//! including publishing, downloading, modifying, and deleting packages.

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use torsh_core::error::{Result, TorshError};

/// Permission for package operations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    /// Read package metadata
    ReadMetadata,
    /// Download package
    Download,
    /// Publish new package
    Publish,
    /// Update existing package
    Update,
    /// Delete package
    Delete,
    /// Manage package versions
    ManageVersions,
    /// Yank package version
    Yank,
    /// Un-yank package version
    Unyank,
    /// Manage package owners
    ManageOwners,
    /// View package statistics
    ViewStats,
    /// Modify package security settings
    ManageSecurity,
    /// Custom permission
    Custom(String),
}

/// Role with a set of permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    /// Role name
    pub name: String,
    /// Role description
    pub description: String,
    /// Permissions granted to this role
    pub permissions: HashSet<Permission>,
    /// Whether this is a built-in role
    pub builtin: bool,
}

/// User identity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    /// User ID (unique identifier)
    pub id: String,
    /// Username
    pub username: String,
    /// Email address
    pub email: String,
    /// Roles assigned to this user
    pub roles: HashSet<String>,
    /// Whether the user is active
    pub active: bool,
    /// When the user was created
    pub created_at: DateTime<Utc>,
}

/// Organization for package management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Organization {
    /// Organization ID
    pub id: String,
    /// Organization name
    pub name: String,
    /// Organization description
    pub description: Option<String>,
    /// Members of the organization
    pub members: HashMap<String, OrganizationMembership>,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
}

/// Organization membership
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganizationMembership {
    /// User ID
    pub user_id: String,
    /// Roles within the organization
    pub roles: HashSet<String>,
    /// When the user joined
    pub joined_at: DateTime<Utc>,
}

/// Package ownership
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageOwnership {
    /// Package name
    pub package_name: String,
    /// Owners (user IDs or organization IDs)
    pub owners: HashSet<String>,
    /// Permissions for specific users
    pub user_permissions: HashMap<String, HashSet<Permission>>,
    /// Public access level
    pub public_access: AccessLevel,
}

/// Access level for public packages
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessLevel {
    /// Public - anyone can read
    Public,
    /// Restricted - only authenticated users can read
    Restricted,
    /// Private - only owners can access
    Private,
}

/// Access control list (ACL) entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AclEntry {
    /// Principal (user or organization ID)
    pub principal: String,
    /// Permissions granted
    pub permissions: HashSet<Permission>,
    /// Expiration time (if any)
    pub expires_at: Option<DateTime<Utc>>,
}

/// Access control manager
pub struct AccessControlManager {
    /// Role definitions
    roles: HashMap<String, Role>,
    /// Users
    users: HashMap<String, User>,
    /// Organizations
    organizations: HashMap<String, Organization>,
    /// Package ownership
    package_ownership: HashMap<String, PackageOwnership>,
}

/// Access check result
#[derive(Debug, Clone)]
pub struct AccessCheckResult {
    /// Whether access is granted
    pub granted: bool,
    /// Reason for denial (if not granted)
    pub denial_reason: Option<String>,
    /// Permissions that were checked
    pub checked_permissions: HashSet<Permission>,
}

impl Role {
    /// Create a new role
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            permissions: HashSet::new(),
            builtin: false,
        }
    }

    /// Add a permission to the role
    pub fn add_permission(&mut self, permission: Permission) {
        self.permissions.insert(permission);
    }

    /// Check if role has a permission
    pub fn has_permission(&self, permission: &Permission) -> bool {
        self.permissions.contains(permission)
    }

    /// Admin role (all permissions)
    pub fn admin() -> Self {
        let mut role = Self {
            name: "admin".to_string(),
            description: "Administrator with all permissions".to_string(),
            permissions: HashSet::new(),
            builtin: true,
        };

        role.permissions.insert(Permission::ReadMetadata);
        role.permissions.insert(Permission::Download);
        role.permissions.insert(Permission::Publish);
        role.permissions.insert(Permission::Update);
        role.permissions.insert(Permission::Delete);
        role.permissions.insert(Permission::ManageVersions);
        role.permissions.insert(Permission::Yank);
        role.permissions.insert(Permission::Unyank);
        role.permissions.insert(Permission::ManageOwners);
        role.permissions.insert(Permission::ViewStats);
        role.permissions.insert(Permission::ManageSecurity);

        role
    }

    /// Maintainer role (can publish and update)
    pub fn maintainer() -> Self {
        let mut role = Self {
            name: "maintainer".to_string(),
            description: "Maintainer can publish and update packages".to_string(),
            permissions: HashSet::new(),
            builtin: true,
        };

        role.permissions.insert(Permission::ReadMetadata);
        role.permissions.insert(Permission::Download);
        role.permissions.insert(Permission::Publish);
        role.permissions.insert(Permission::Update);
        role.permissions.insert(Permission::ManageVersions);
        role.permissions.insert(Permission::Yank);
        role.permissions.insert(Permission::Unyank);

        role
    }

    /// Contributor role (can only download)
    pub fn contributor() -> Self {
        let mut role = Self {
            name: "contributor".to_string(),
            description: "Contributor can download packages".to_string(),
            permissions: HashSet::new(),
            builtin: true,
        };

        role.permissions.insert(Permission::ReadMetadata);
        role.permissions.insert(Permission::Download);

        role
    }

    /// Viewer role (can only view metadata)
    pub fn viewer() -> Self {
        let mut role = Self {
            name: "viewer".to_string(),
            description: "Viewer can only view metadata".to_string(),
            permissions: HashSet::new(),
            builtin: true,
        };

        role.permissions.insert(Permission::ReadMetadata);

        role
    }
}

impl User {
    /// Create a new user
    pub fn new(id: String, username: String, email: String) -> Self {
        Self {
            id,
            username,
            email,
            roles: HashSet::new(),
            active: true,
            created_at: Utc::now(),
        }
    }

    /// Add a role to the user
    pub fn add_role(&mut self, role: String) {
        self.roles.insert(role);
    }

    /// Check if user has a role
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.contains(role)
    }

    /// Deactivate user
    pub fn deactivate(&mut self) {
        self.active = false;
    }
}

impl Organization {
    /// Create a new organization
    pub fn new(id: String, name: String) -> Self {
        Self {
            id,
            name,
            description: None,
            members: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    /// Add a member to the organization
    pub fn add_member(&mut self, user_id: String, roles: HashSet<String>) {
        let membership = OrganizationMembership {
            user_id: user_id.clone(),
            roles,
            joined_at: Utc::now(),
        };
        self.members.insert(user_id, membership);
    }

    /// Remove a member from the organization
    pub fn remove_member(&mut self, user_id: &str) {
        self.members.remove(user_id);
    }

    /// Check if user is a member
    pub fn is_member(&self, user_id: &str) -> bool {
        self.members.contains_key(user_id)
    }

    /// Get member's roles
    pub fn get_member_roles(&self, user_id: &str) -> Option<&HashSet<String>> {
        self.members.get(user_id).map(|m| &m.roles)
    }
}

impl PackageOwnership {
    /// Create new package ownership
    pub fn new(package_name: String) -> Self {
        Self {
            package_name,
            owners: HashSet::new(),
            user_permissions: HashMap::new(),
            public_access: AccessLevel::Public,
        }
    }

    /// Add an owner
    pub fn add_owner(&mut self, owner_id: String) {
        self.owners.insert(owner_id);
    }

    /// Remove an owner
    pub fn remove_owner(&mut self, owner_id: &str) {
        self.owners.remove(owner_id);
    }

    /// Check if principal is an owner
    pub fn is_owner(&self, principal_id: &str) -> bool {
        self.owners.contains(principal_id)
    }

    /// Grant permission to a user
    pub fn grant_permission(&mut self, user_id: String, permission: Permission) {
        self.user_permissions
            .entry(user_id)
            .or_default()
            .insert(permission);
    }

    /// Revoke permission from a user
    pub fn revoke_permission(&mut self, user_id: &str, permission: &Permission) {
        if let Some(perms) = self.user_permissions.get_mut(user_id) {
            perms.remove(permission);
        }
    }
}

impl Default for AccessControlManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AccessControlManager {
    /// Create a new access control manager
    pub fn new() -> Self {
        let mut manager = Self {
            roles: HashMap::new(),
            users: HashMap::new(),
            organizations: HashMap::new(),
            package_ownership: HashMap::new(),
        };

        // Register built-in roles
        manager.register_role(Role::admin());
        manager.register_role(Role::maintainer());
        manager.register_role(Role::contributor());
        manager.register_role(Role::viewer());

        manager
    }

    /// Register a role
    pub fn register_role(&mut self, role: Role) {
        self.roles.insert(role.name.clone(), role);
    }

    /// Create a user
    pub fn create_user(&mut self, id: String, username: String, email: String) -> Result<User> {
        if self.users.contains_key(&id) {
            return Err(TorshError::InvalidArgument(format!(
                "User already exists: {}",
                id
            )));
        }

        let user = User::new(id.clone(), username, email);
        self.users.insert(id, user.clone());
        Ok(user)
    }

    /// Get a user
    pub fn get_user(&self, user_id: &str) -> Option<&User> {
        self.users.get(user_id)
    }

    /// Create an organization
    pub fn create_organization(&mut self, id: String, name: String) -> Result<Organization> {
        if self.organizations.contains_key(&id) {
            return Err(TorshError::InvalidArgument(format!(
                "Organization already exists: {}",
                id
            )));
        }

        let org = Organization::new(id.clone(), name);
        self.organizations.insert(id, org.clone());
        Ok(org)
    }

    /// Get an organization
    pub fn get_organization(&self, org_id: &str) -> Option<&Organization> {
        self.organizations.get(org_id)
    }

    /// Set package ownership
    pub fn set_package_ownership(&mut self, ownership: PackageOwnership) {
        self.package_ownership
            .insert(ownership.package_name.clone(), ownership);
    }

    /// Check if user has permission for a package operation
    pub fn check_access(
        &self,
        user_id: &str,
        package_name: &str,
        permission: &Permission,
    ) -> AccessCheckResult {
        // Get user
        let user = match self.get_user(user_id) {
            Some(u) => u,
            None => {
                return AccessCheckResult {
                    granted: false,
                    denial_reason: Some("User not found".to_string()),
                    checked_permissions: HashSet::from([permission.clone()]),
                };
            }
        };

        // Check if user is active
        if !user.active {
            return AccessCheckResult {
                granted: false,
                denial_reason: Some("User is not active".to_string()),
                checked_permissions: HashSet::from([permission.clone()]),
            };
        }

        // Get package ownership
        let ownership = match self.package_ownership.get(package_name) {
            Some(o) => o,
            None => {
                return AccessCheckResult {
                    granted: false,
                    denial_reason: Some("Package not found".to_string()),
                    checked_permissions: HashSet::from([permission.clone()]),
                };
            }
        };

        // Check if user is owner
        if ownership.is_owner(user_id) {
            return AccessCheckResult {
                granted: true,
                denial_reason: None,
                checked_permissions: HashSet::from([permission.clone()]),
            };
        }

        // Check user-specific permissions
        if let Some(perms) = ownership.user_permissions.get(user_id) {
            if perms.contains(permission) {
                return AccessCheckResult {
                    granted: true,
                    denial_reason: None,
                    checked_permissions: HashSet::from([permission.clone()]),
                };
            }
        }

        // Check role-based permissions
        for role_name in &user.roles {
            if let Some(role) = self.roles.get(role_name) {
                if role.has_permission(permission) {
                    return AccessCheckResult {
                        granted: true,
                        denial_reason: None,
                        checked_permissions: HashSet::from([permission.clone()]),
                    };
                }
            }
        }

        // Check public access for read operations
        if ownership.public_access == AccessLevel::Public
            && (permission == &Permission::ReadMetadata || permission == &Permission::Download)
        {
            return AccessCheckResult {
                granted: true,
                denial_reason: None,
                checked_permissions: HashSet::from([permission.clone()]),
            };
        }

        AccessCheckResult {
            granted: false,
            denial_reason: Some("Insufficient permissions".to_string()),
            checked_permissions: HashSet::from([permission.clone()]),
        }
    }

    /// Grant role to user
    pub fn grant_role(&mut self, user_id: &str, role_name: &str) -> Result<()> {
        // Check if role exists
        if !self.roles.contains_key(role_name) {
            return Err(TorshError::InvalidArgument(format!(
                "Role not found: {}",
                role_name
            )));
        }

        // Get user and add role
        let user = self
            .users
            .get_mut(user_id)
            .ok_or_else(|| TorshError::InvalidArgument(format!("User not found: {}", user_id)))?;

        user.add_role(role_name.to_string());
        Ok(())
    }

    /// Revoke role from user
    pub fn revoke_role(&mut self, user_id: &str, role_name: &str) -> Result<()> {
        let user = self
            .users
            .get_mut(user_id)
            .ok_or_else(|| TorshError::InvalidArgument(format!("User not found: {}", user_id)))?;

        user.roles.remove(role_name);
        Ok(())
    }

    /// Add user to organization
    pub fn add_to_organization(
        &mut self,
        user_id: &str,
        org_id: &str,
        roles: HashSet<String>,
    ) -> Result<()> {
        // Verify user exists
        if !self.users.contains_key(user_id) {
            return Err(TorshError::InvalidArgument(format!(
                "User not found: {}",
                user_id
            )));
        }

        // Get organization and add member
        let org = self.organizations.get_mut(org_id).ok_or_else(|| {
            TorshError::InvalidArgument(format!("Organization not found: {}", org_id))
        })?;

        org.add_member(user_id.to_string(), roles);
        Ok(())
    }
}

impl AccessCheckResult {
    /// Create a granted result
    pub fn granted() -> Self {
        Self {
            granted: true,
            denial_reason: None,
            checked_permissions: HashSet::new(),
        }
    }

    /// Create a denied result
    pub fn denied(reason: String) -> Self {
        Self {
            granted: false,
            denial_reason: Some(reason),
            checked_permissions: HashSet::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_creation() {
        let mut role = Role::new("test-role".to_string(), "Test role".to_string());
        assert_eq!(role.name, "test-role");

        role.add_permission(Permission::ReadMetadata);
        assert!(role.has_permission(&Permission::ReadMetadata));
        assert!(!role.has_permission(&Permission::Publish));
    }

    #[test]
    fn test_builtin_roles() {
        let admin = Role::admin();
        assert!(admin.has_permission(&Permission::Delete));
        assert!(admin.builtin);

        let viewer = Role::viewer();
        assert!(viewer.has_permission(&Permission::ReadMetadata));
        assert!(!viewer.has_permission(&Permission::Publish));
    }

    #[test]
    fn test_user_creation() {
        let mut user = User::new(
            "user1".to_string(),
            "testuser".to_string(),
            "test@example.com".to_string(),
        );

        assert_eq!(user.username, "testuser");
        assert!(user.active);

        user.add_role("admin".to_string());
        assert!(user.has_role("admin"));

        user.deactivate();
        assert!(!user.active);
    }

    #[test]
    fn test_organization() {
        let mut org = Organization::new("org1".to_string(), "Test Org".to_string());

        let roles = HashSet::from(["maintainer".to_string()]);
        org.add_member("user1".to_string(), roles);

        assert!(org.is_member("user1"));
        assert!(!org.is_member("user2"));

        org.remove_member("user1");
        assert!(!org.is_member("user1"));
    }

    #[test]
    fn test_package_ownership() {
        let mut ownership = PackageOwnership::new("test-package".to_string());

        ownership.add_owner("user1".to_string());
        assert!(ownership.is_owner("user1"));
        assert!(!ownership.is_owner("user2"));

        ownership.grant_permission("user2".to_string(), Permission::Download);
        assert!(ownership
            .user_permissions
            .get("user2")
            .unwrap()
            .contains(&Permission::Download));

        ownership.revoke_permission("user2", &Permission::Download);
        assert!(!ownership
            .user_permissions
            .get("user2")
            .unwrap()
            .contains(&Permission::Download));
    }

    #[test]
    fn test_access_control_manager() {
        let mut acl = AccessControlManager::new();

        // Create user
        let user = acl
            .create_user(
                "user1".to_string(),
                "testuser".to_string(),
                "test@example.com".to_string(),
            )
            .unwrap();
        assert_eq!(user.username, "testuser");

        // Create organization
        let org = acl
            .create_organization("org1".to_string(), "Test Org".to_string())
            .unwrap();
        assert_eq!(org.name, "Test Org");

        // Grant role
        acl.grant_role("user1", "maintainer").unwrap();
        assert!(acl.get_user("user1").unwrap().has_role("maintainer"));
    }

    #[test]
    fn test_access_check() {
        let mut acl = AccessControlManager::new();

        // Create user
        acl.create_user(
            "user1".to_string(),
            "testuser".to_string(),
            "test@example.com".to_string(),
        )
        .unwrap();

        // Set up package ownership
        let mut ownership = PackageOwnership::new("test-package".to_string());
        ownership.public_access = AccessLevel::Public;
        acl.set_package_ownership(ownership);

        // Check read access (should be granted for public package)
        let result = acl.check_access("user1", "test-package", &Permission::ReadMetadata);
        assert!(result.granted);

        // Check publish access (should be denied)
        let result = acl.check_access("user1", "test-package", &Permission::Publish);
        assert!(!result.granted);
    }

    #[test]
    fn test_owner_access() {
        let mut acl = AccessControlManager::new();

        // Create user
        acl.create_user(
            "user1".to_string(),
            "testuser".to_string(),
            "test@example.com".to_string(),
        )
        .unwrap();

        // Set up package ownership with user1 as owner
        let mut ownership = PackageOwnership::new("test-package".to_string());
        ownership.add_owner("user1".to_string());
        acl.set_package_ownership(ownership);

        // Check publish access (should be granted for owner)
        let result = acl.check_access("user1", "test-package", &Permission::Publish);
        assert!(result.granted);

        // Check delete access (should be granted for owner)
        let result = acl.check_access("user1", "test-package", &Permission::Delete);
        assert!(result.granted);
    }
}
