function [vx, ax, vy, ay] = gibmirpars(sx,sy,vmaxx,vmaxy,amaxx,amaxy,tj)


% Detektion der Achse mit geringerer Bewegungszeit bei vollen Achsparametern

sx=abs(sx);
sy=abs(sy);

tx=emotion(sx,vmaxx,amaxx,tj);
ty=emotion(sy,vmaxy,amaxy,tj);

% Berechnung der neuen Fahrparameter

if ty > tx  % Parameter der X-Achse müssen runterskalliert werden
    
    vy1=vmaxy;   % keine Änderung notwendig
    ay1=amaxy;   % keine Änderung notwendig
    
    v=sx/sy;     % Verhältnis der Wege
    vx1=vy1*v;   % entsprechend tm=f(s,v,a,tj)=f(s*v,v*v,a*v,tj)
    ax1=ay1*v;
    
    % Pruefung auf Überschreitung der Achslimits und Skallierung
    
    w=0;
    x=0;
    y=0;
    
    if vx1 > vmaxx
        w=vx1/vmaxx;
        %vx1=vx1/w;
    end
    
    if ax1 > amaxx
        x=ax1/amaxx; 
        %ax1=ax1/x;
    end
        
    if w > 0 || x > 0
       y=max(w,x);
       vx1=vx1/y;
       vy1=vy1/y;
       ax1=ax1/y;
       ay1=ay1/y;
    end
    
    
else        % Parameter der Y-Achse müssen runterskalliert werden
    
    vx1=vmaxx;   % keine Änderung notwendig
    ax1=amaxx;   % keine Änderung notwendig
    
    v=sy/sx;     % Verhältnis der Wege
    
    vy1=vx1*v;   % entsprechend tm=f(s,v,a,tj)=f(s*v,v*v,a*v,tj)
    ay1=ax1*v;

    % Pruefung auf Überschreitung der Achslimits und Skallierung
    
    w=0;
    x=0;
    y=0;
    
    if vy1 > vmaxy
        w=vy1/vmaxy;
        %vy1=vy1/w;
    end
    
    if ay1 > amaxy
        x=ay1/amaxy; 
        %ay1=ay1/x;
    end
        
    if w > 0 | x > 0
       y=max(w,x);
       vx1=vx1/y;
       vy1=vy1/y;
       ax1=ax1/y;
       ay1=ay1/y;
    end
end

vx=vx1
vy=vy1
ax=ax1
ay=ay1
