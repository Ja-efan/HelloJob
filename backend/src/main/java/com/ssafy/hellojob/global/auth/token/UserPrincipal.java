package com.ssafy.hellojob.global.auth.token;

import com.ssafy.hellojob.domain.user.entity.User;
import lombok.Getter;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.oauth2.core.user.OAuth2User;

import java.util.Collection;
import java.util.Collections;
import java.util.Map;

@Getter
public class UserPrincipal implements OAuth2User {

    private final User user;
    private final Collection<? extends GrantedAuthority> authorities;
    private final OAuth2User oauth2User;

    // JWT 인증용 생성자
    public UserPrincipal(Collection<? extends GrantedAuthority> authorities, User user) {
        this.authorities = authorities;
        this.user = user;
        this.oauth2User = null;
    }

    // OAuth2 인증용 생성자
    public UserPrincipal(OAuth2User oauth2User, User user) {
        this.oauth2User = oauth2User;
        this.user = user;
        this.authorities = oauth2User.getAuthorities();
    }

    @Override
    public Map<String, Object> getAttributes() {
        return oauth2User != null ? oauth2User.getAttributes() : Collections.emptyMap();
    }

    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        return authorities;
    }

    @Override
    public String getName() {
        return oauth2User != null ? oauth2User.getName() : user.getNickname();
    }

    public String getEmail() {
        return user.getEmail();
    }

    public Integer getUserId() {
        return user.getUserId();
    }

    public String getNickname() { return user.getNickname(); }

    public Boolean getWithdraw() { return user.getWithdraw(); }

}