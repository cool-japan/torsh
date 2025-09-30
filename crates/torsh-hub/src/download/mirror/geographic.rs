//! Geographic Calculations and Proximity Optimization
//!
//! This module provides geographic calculation capabilities for mirror management,
//! including distance calculations, proximity-based scoring, and location-based
//! optimization algorithms for intelligent mirror selection.

use super::types::*;
use std::collections::HashMap;

// ================================================================================================
// Geographic Calculator Implementation
// ================================================================================================

impl GeographicCalculator {
    /// Create a new geographic calculator
    pub fn new() -> Self {
        Self {
            enabled: true,
            user_location: None,
            distance_cache: HashMap::new(),
        }
    }

    /// Create a new geographic calculator with specified user location
    pub fn with_user_location(latitude: f64, longitude: f64, estimated: bool) -> Self {
        Self {
            enabled: true,
            user_location: Some(UserLocation {
                latitude,
                longitude,
                estimated,
            }),
            distance_cache: HashMap::new(),
        }
    }

    /// Set the user location for geographic calculations
    pub fn set_user_location(&mut self, latitude: f64, longitude: f64, estimated: bool) {
        self.user_location = Some(UserLocation {
            latitude,
            longitude,
            estimated,
        });
        // Clear cache when location changes
        self.distance_cache.clear();
    }

    /// Estimate user location based on IP geolocation or other methods
    pub fn estimate_user_location(&mut self) -> Option<&UserLocation> {
        // In a real implementation, this would use IP geolocation services
        // For now, we'll set a default location (US East Coast)
        if self.user_location.is_none() {
            self.set_user_location(39.0438, -77.4874, true); // Ashburn, VA
        }
        self.user_location.as_ref()
    }

    /// Calculate geographic proximity score for a mirror
    ///
    /// Returns a score from 0.0 to 1.0 where 1.0 indicates closest proximity.
    /// Takes into account geographic distance and regional preferences.
    pub fn calculate_geographic_score(&self, mirror: &MirrorServer) -> f64 {
        if !self.enabled {
            return 0.5; // Neutral score if geographic optimization disabled
        }

        if let Some(user_location) = &self.user_location {
            if let (Some(mirror_lat), Some(mirror_lon)) =
                (mirror.location.latitude, mirror.location.longitude)
            {
                let distance = self.calculate_distance(
                    user_location.latitude,
                    user_location.longitude,
                    mirror_lat,
                    mirror_lon,
                );

                // Convert distance to score (closer is better)
                // Assume max useful distance of 20,000 km (half Earth circumference)
                let max_distance = 20000.0;
                return (max_distance - distance.min(max_distance)) / max_distance;
            }
        }

        // Fallback: Use simplified geographic scoring based on common regions
        self.calculate_regional_score(&mirror.location.country)
    }

    /// Calculate distance between two geographic points using Haversine formula
    ///
    /// # Arguments
    /// * `lat1` - Latitude of first point in degrees
    /// * `lon1` - Longitude of first point in degrees
    /// * `lat2` - Latitude of second point in degrees
    /// * `lon2` - Longitude of second point in degrees
    ///
    /// # Returns
    /// Distance in kilometers
    pub fn calculate_distance(&self, lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
        // Check cache first
        let cache_key = format!("{:.4},{:.4},{:.4},{:.4}", lat1, lon1, lat2, lon2);
        if let Some(&cached_distance) = self.distance_cache.get(&cache_key) {
            return cached_distance;
        }

        const R: f64 = 6371.0; // Earth's radius in kilometers

        let lat1_rad = lat1.to_radians();
        let lat2_rad = lat2.to_radians();
        let delta_lat = (lat2 - lat1).to_radians();
        let delta_lon = (lon2 - lon1).to_radians();

        let a = (delta_lat / 2.0).sin().powi(2)
            + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

        let distance = R * c;

        // Note: We can't cache here since this method takes &self
        // Caching would be handled by the caller if needed
        distance
    }

    /// Calculate distance with caching support
    pub fn calculate_distance_cached(&mut self, lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
        let cache_key = format!("{:.4},{:.4},{:.4},{:.4}", lat1, lon1, lat2, lon2);

        if let Some(&cached_distance) = self.distance_cache.get(&cache_key) {
            return cached_distance;
        }

        let distance = self.calculate_distance(lat1, lon1, lat2, lon2);
        self.distance_cache.insert(cache_key, distance);
        distance
    }

    /// Sort mirrors by geographic proximity to user location
    ///
    /// Returns mirrors sorted by proximity (closest first). Mirrors without
    /// geographic coordinates are placed at the end with regional scoring.
    pub fn sort_by_geographic_proximity(
        &self,
        mut mirrors: Vec<MirrorServer>,
    ) -> Vec<MirrorServer> {
        if !self.enabled {
            return mirrors; // Return unsorted if geographic optimization disabled
        }

        if let Some(user_location) = &self.user_location {
            mirrors.sort_by(|a, b| {
                let score_a = self.calculate_mirror_proximity_score(a, user_location);
                let score_b = self.calculate_mirror_proximity_score(b, user_location);

                // Sort by score descending (higher score = closer proximity)
                score_b
                    .partial_cmp(&score_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            // Sort by regional preference if no user location available
            mirrors.sort_by(|a, b| {
                let score_a = self.calculate_regional_score(&a.location.country);
                let score_b = self.calculate_regional_score(&b.location.country);
                score_b
                    .partial_cmp(&score_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        mirrors
    }

    /// Calculate proximity score for a specific mirror relative to user location
    fn calculate_mirror_proximity_score(
        &self,
        mirror: &MirrorServer,
        user_location: &UserLocation,
    ) -> f64 {
        if let (Some(mirror_lat), Some(mirror_lon)) =
            (mirror.location.latitude, mirror.location.longitude)
        {
            let distance = self.calculate_distance(
                user_location.latitude,
                user_location.longitude,
                mirror_lat,
                mirror_lon,
            );

            // Convert distance to score (closer is better)
            let max_distance = 20000.0; // Half Earth circumference
            (max_distance - distance.min(max_distance)) / max_distance
        } else {
            // Fallback to regional scoring for mirrors without coordinates
            self.calculate_regional_score(&mirror.location.country)
        }
    }

    /// Calculate regional preference score based on country codes
    ///
    /// This provides a fallback scoring system when precise geographic
    /// coordinates are not available.
    fn calculate_regional_score(&self, country_code: &str) -> f64 {
        // Default scoring assumes US-based users
        // In a real implementation, this would be more sophisticated
        match country_code {
            "US" => 1.0,                              // Highest preference for US
            "CA" => 0.9,                              // High preference for Canada (close to US)
            "MX" => 0.85,                             // Mexico (North America)
            "GB" | "IE" => 0.8,                       // UK and Ireland (English-speaking)
            "DE" | "FR" | "NL" | "BE" | "CH" => 0.75, // Western Europe
            "IT" | "ES" | "PT" => 0.7,                // Southern Europe
            "SE" | "NO" | "DK" | "FI" => 0.7,         // Nordic countries
            "PL" | "CZ" | "HU" | "AT" => 0.65,        // Central Europe
            "JP" | "KR" => 0.6,                       // East Asia (good infrastructure)
            "SG" | "HK" | "TW" => 0.6,                // Southeast Asia hubs
            "AU" | "NZ" => 0.55,                      // Oceania
            "BR" | "AR" | "CL" => 0.5,                // South America
            "IN" | "TH" | "MY" => 0.45,               // South/Southeast Asia
            "CN" => 0.4,                              // China (may have connectivity issues)
            "RU" => 0.35,                             // Russia
            "ZA" => 0.3,                              // Africa
            _ => 0.25,                                // Other/unknown regions
        }
    }

    /// Get estimated latency based on geographic distance
    ///
    /// Provides a rough estimate of network latency based on geographic distance.
    /// This is useful for initial estimates before actual measurements are available.
    pub fn estimate_latency_from_distance(&self, distance_km: f64) -> u64 {
        // Speed of light in optical fiber is approximately 200,000 km/s
        // Add routing overhead and processing delays
        let base_latency = (distance_km / 200.0) * 2.0; // Round trip
        let routing_overhead = distance_km * 0.01; // 1% routing overhead per 100km
        let processing_delay = 10.0; // Base processing delay in ms

        (base_latency + routing_overhead + processing_delay) as u64
    }

    /// Calculate network topology score based on connectivity patterns
    ///
    /// This considers factors like internet exchange points, submarine cables,
    /// and typical routing patterns that affect real-world connectivity.
    pub fn calculate_network_topology_score(&self, mirror: &MirrorServer) -> f64 {
        let mut score = 0.5; // Base score

        // Bonus for mirrors in major internet hubs
        match mirror.location.city.as_str() {
            "Ashburn" | "Santa Clara" | "Seattle" => score += 0.3, // Major US hubs
            "London" | "Amsterdam" | "Frankfurt" => score += 0.25, // Major EU hubs
            "Singapore" | "Hong Kong" | "Tokyo" => score += 0.2,   // Major APAC hubs
            "Sydney" | "Mumbai" | "São Paulo" => score += 0.15,    // Regional hubs
            _ => {}
        }

        // Bonus for high-quality network providers
        match mirror.provider_info.name.as_str() {
            "AWS" | "Google Cloud" | "Microsoft Azure" => score += 0.2,
            "Cloudflare" | "Fastly" | "Akamai" => score += 0.25, // CDN providers
            "Level3" | "Cogent" | "Hurricane Electric" => score += 0.15, // Transit providers
            _ => {}
        }

        // Consider network quality metrics
        score += mirror.provider_info.network_quality.quality_score * 0.2;

        // Clamp to valid range
        score.min(1.0).max(0.0)
    }

    /// Find optimal mirrors for a given region
    ///
    /// Returns a list of mirrors optimized for serving a specific geographic region,
    /// considering both proximity and network topology.
    pub fn find_optimal_mirrors_for_region(
        &self,
        mirrors: &[MirrorServer],
        target_country: &str,
        max_mirrors: usize,
    ) -> Vec<MirrorServer> {
        let mut scored_mirrors: Vec<(MirrorServer, f64)> = mirrors
            .iter()
            .map(|mirror| {
                let mut score = 0.0;

                // Perfect score for same country
                if mirror.location.country == target_country {
                    score += 1.0;
                } else {
                    // Regional proximity scoring
                    score += self.calculate_regional_score(&mirror.location.country);
                }

                // Add network topology bonus
                score += self.calculate_network_topology_score(mirror) * 0.3;

                // Add reliability bonus
                score += mirror.reliability_score * 0.2;

                (mirror.clone(), score)
            })
            .collect();

        // Sort by score (highest first)
        scored_mirrors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top mirrors
        scored_mirrors
            .into_iter()
            .take(max_mirrors)
            .map(|(mirror, _)| mirror)
            .collect()
    }

    /// Clear the distance calculation cache
    pub fn clear_cache(&mut self) {
        self.distance_cache.clear();
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> (usize, usize) {
        (self.distance_cache.len(), self.distance_cache.capacity())
    }

    /// Enable or disable geographic optimization
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.clear_cache();
        }
    }

    /// Check if geographic optimization is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get current user location
    pub fn get_user_location(&self) -> Option<&UserLocation> {
        self.user_location.as_ref()
    }
}

// ================================================================================================
// Geographic Utility Functions
// ================================================================================================

/// Calculate the midpoint between multiple geographic locations
///
/// Useful for finding optimal placement for new mirrors or estimating
/// central locations for regional deployments.
pub fn calculate_geographic_midpoint(locations: &[(f64, f64)]) -> Option<(f64, f64)> {
    if locations.is_empty() {
        return None;
    }

    let mut total_lat = 0.0;
    let mut total_lon = 0.0;

    for &(lat, lon) in locations {
        total_lat += lat;
        total_lon += lon;
    }

    Some((
        total_lat / locations.len() as f64,
        total_lon / locations.len() as f64,
    ))
}

/// Determine the continent for a given country code
pub fn get_continent_for_country(country_code: &str) -> &'static str {
    match country_code {
        "US" | "CA" | "MX" | "GT" | "BZ" | "SV" | "HN" | "NI" | "CR" | "PA" => "North America",
        "BR" | "AR" | "CL" | "PE" | "CO" | "VE" | "EC" | "BO" | "PY" | "UY" | "GY" | "SR"
        | "GF" => "South America",
        "GB" | "DE" | "FR" | "IT" | "ES" | "NL" | "BE" | "CH" | "AT" | "SE" | "NO" | "DK"
        | "FI" | "PL" | "CZ" | "HU" | "SK" | "SI" | "HR" | "RS" | "BG" | "RO" | "GR" | "PT"
        | "IE" | "LU" | "LI" | "MC" | "AD" | "SM" | "VA" | "MT" | "CY" => "Europe",
        "CN" | "JP" | "KR" | "TW" | "HK" | "MO" | "MN" => "East Asia",
        "IN" | "PK" | "BD" | "LK" | "NP" | "BT" | "MV" | "AF" => "South Asia",
        "TH" | "MY" | "SG" | "ID" | "PH" | "VN" | "LA" | "KH" | "MM" | "BN" | "TL" => {
            "Southeast Asia"
        }
        "RU" | "KZ" | "UZ" | "TM" | "TJ" | "KG" | "AM" | "AZ" | "GE" => "Central Asia",
        "TR" | "SA" | "AE" | "QA" | "KW" | "BH" | "OM" | "YE" | "IQ" | "IR" | "IL" | "PS"
        | "JO" | "LB" | "SY" => "Middle East",
        "AU" | "NZ" | "FJ" | "PG" | "VU" | "SB" | "NC" | "PF" => "Oceania",
        "EG" | "ZA" | "NG" | "KE" | "MA" | "TN" | "DZ" | "LY" | "ET" | "GH" | "UG" | "TZ"
        | "MZ" | "ZW" | "ZM" | "MW" | "BW" | "NA" | "SZ" | "LS" => "Africa",
        _ => "Unknown",
    }
}

/// Validate geographic coordinates
pub fn validate_coordinates(latitude: f64, longitude: f64) -> bool {
    latitude >= -90.0 && latitude <= 90.0 && longitude >= -180.0 && longitude <= 180.0
}

/// Normalize longitude to [-180, 180] range
pub fn normalize_longitude(longitude: f64) -> f64 {
    let normalized = (longitude + 180.0) % 360.0;
    if normalized < 0.0 {
        normalized + 360.0 - 180.0
    } else {
        normalized - 180.0
    }
}

/// Calculate the bounding box for a set of coordinates with padding
pub fn calculate_bounding_box(
    coordinates: &[(f64, f64)],
    padding_km: f64,
) -> Option<(f64, f64, f64, f64)> {
    if coordinates.is_empty() {
        return None;
    }

    let mut min_lat = f64::MAX;
    let mut max_lat = f64::MIN;
    let mut min_lon = f64::MAX;
    let mut max_lon = f64::MIN;

    for &(lat, lon) in coordinates {
        min_lat = min_lat.min(lat);
        max_lat = max_lat.max(lat);
        min_lon = min_lon.min(lon);
        max_lon = max_lon.max(lon);
    }

    // Convert padding from km to degrees (rough approximation)
    let lat_padding = padding_km / 111.0; // ~111 km per degree latitude
    let lon_padding = padding_km / (111.0 * ((min_lat + max_lat) / 2.0).to_radians().cos());

    Some((
        min_lat - lat_padding,
        min_lon - lon_padding,
        max_lat + lat_padding,
        max_lon + lon_padding,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_calculation() {
        let calculator = GeographicCalculator::new();

        // Test distance between New York and London (approximately 5585 km)
        let distance = calculator.calculate_distance(40.7128, -74.0060, 51.5074, -0.1278);
        assert!(
            (distance - 5585.0).abs() < 50.0,
            "Distance calculation should be accurate within 50km"
        );
    }

    #[test]
    fn test_geographic_score_calculation() {
        let mut calculator = GeographicCalculator::new();
        calculator.set_user_location(40.7128, -74.0060, false); // New York

        let mirror = create_mirror_server(
            "test",
            "https://test.example.com",
            "US",
            "New York",
            "TestProvider",
        );

        let score = calculator.calculate_geographic_score(&mirror);
        assert!(
            score >= 0.0 && score <= 1.0,
            "Geographic score should be between 0 and 1"
        );
    }

    #[test]
    fn test_regional_score_calculation() {
        let calculator = GeographicCalculator::new();

        assert_eq!(calculator.calculate_regional_score("US"), 1.0);
        assert_eq!(calculator.calculate_regional_score("CA"), 0.9);
        assert!(
            calculator.calculate_regional_score("DE") > calculator.calculate_regional_score("CN")
        );
    }

    #[test]
    fn test_coordinate_validation() {
        assert!(validate_coordinates(40.7128, -74.0060)); // Valid NYC coordinates
        assert!(!validate_coordinates(91.0, 0.0)); // Invalid latitude
        assert!(!validate_coordinates(0.0, 181.0)); // Invalid longitude
    }

    #[test]
    fn test_longitude_normalization() {
        assert_eq!(normalize_longitude(181.0), -179.0);
        assert_eq!(normalize_longitude(-181.0), 179.0);
        assert_eq!(normalize_longitude(0.0), 0.0);
    }

    #[test]
    fn test_continent_detection() {
        assert_eq!(get_continent_for_country("US"), "North America");
        assert_eq!(get_continent_for_country("DE"), "Europe");
        assert_eq!(get_continent_for_country("JP"), "East Asia");
        assert_eq!(get_continent_for_country("AU"), "Oceania");
    }

    #[test]
    fn test_geographic_midpoint() {
        let locations = vec![(40.0, -74.0), (41.0, -73.0)];
        let midpoint = calculate_geographic_midpoint(&locations).unwrap();
        assert_eq!(midpoint, (40.5, -73.5));
    }

    #[test]
    fn test_bounding_box_calculation() {
        let coordinates = vec![(40.0, -74.0), (41.0, -73.0)];
        let bbox = calculate_bounding_box(&coordinates, 10.0).unwrap();

        // Should expand beyond the original coordinates
        assert!(bbox.0 < 40.0); // min_lat
        assert!(bbox.1 < -74.0); // min_lon
        assert!(bbox.2 > 41.0); // max_lat
        assert!(bbox.3 > -73.0); // max_lon
    }
}
