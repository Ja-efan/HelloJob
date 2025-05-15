package com.ssafy.hellojob.global.util;

import com.ssafy.hellojob.global.auth.dto.TokenDto;
import com.ssafy.hellojob.global.exception.BaseException;
import io.jsonwebtoken.*;
import io.jsonwebtoken.io.Decoders;
import io.jsonwebtoken.security.Keys;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import javax.crypto.SecretKey;
import java.util.Arrays;
import java.util.Date;

import static com.ssafy.hellojob.global.exception.ErrorCode.EXPIRED_TOKEN;
import static com.ssafy.hellojob.global.exception.ErrorCode.INVALID_TOKEN;

@Component
public class JwtUtil {

    private final SecretKey key;
    private final long accessTokenExpiration;
    private final long refreshTokenExpiration;

    public JwtUtil(
            @Value("${jwt.secret}") String secret,
            @Value("${jwt.access-token-expiration}") long accessTokenExpiration,
            @Value("${jwt.refresh-token-expiration}") long refreshTokenExpiration
    ) {
        byte[] keyBytes = Decoders.BASE64.decode(secret);
        this.key = Keys.hmacShaKeyFor(keyBytes);
        this.accessTokenExpiration = accessTokenExpiration;
        this.refreshTokenExpiration = refreshTokenExpiration;
    }

    public TokenDto generateTokens(String email) {
        String accessToken = generateToken(email, accessTokenExpiration);
        String refreshToken = generateToken(email, refreshTokenExpiration);
        return new TokenDto(accessToken, refreshToken, accessTokenExpiration);
    }

    private String generateToken(String email, long expiration) {
        Date now = new Date();
        Date expiryDate = new Date(now.getTime() + expiration);

        return Jwts.builder()
                .subject(email)
                .issuedAt(now)
                .expiration(expiryDate)
                .signWith(key)
                .compact();
    }

    public String getEmailFromToken(String token) {
        return Jwts.parser()
                .verifyWith(key)
                .build()
                .parseSignedClaims(token)
                .getPayload()
                .getSubject();
    }

    private Claims extractAllClaims(String token) {
        return Jwts.parser()
                .verifyWith(key)                // setSigningKey() 대신 verifyWith()
                .build()
                .parseSignedClaims(token)      // parseClaimsJws() 대신 parseSignedClaims()
                .getPayload();                 // getBody() 대신 getPayload()
    }

    public String extractAccessToken(HttpServletRequest request) {

        if (request.getCookies() == null)
            return null;

        return Arrays.stream(request.getCookies())
                .filter(cookie -> "access_token".equals(cookie.getName()))
                .map(Cookie::getValue)
                .findFirst()
                .orElse(null);
    }

    public String extractRefreshToken(HttpServletRequest request) {

        if (request.getCookies() == null)
            return null;

        return Arrays.stream(request.getCookies())
                .filter(cookie -> "refresh_token".equals(cookie.getName()))
                .map(Cookie::getValue)
                .findFirst()
                .orElse(null);
    }

    public boolean validateToken(String token) {
        try {
            extractAllClaims(token);
            return true;
        } catch (SecurityException | MalformedJwtException |
                 UnsupportedJwtException | IllegalArgumentException e) {
            throw new BaseException(INVALID_TOKEN);
        } catch (ExpiredJwtException e) {
            throw new BaseException(EXPIRED_TOKEN);
        }
    }

}